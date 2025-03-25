#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
//gcc -o 1 -lm 1.c && ./1 lena512.bmp lena512_filtered.bmp
// Gaussian function
double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// Manual bilateral filter
void bilateral_filter(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int radius = d / 2;

    // Precompute spatial Gaussian weights
    double *spatial_weights = (double *)malloc(d * d * sizeof(double));
    if (!spatial_weights) {
        printf("Memory allocation for spatial weights failed!\n");
        return;
    }

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int x = i - radius, y = j - radius;
            spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
        }
    }

    // Process image
    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
            double weight_sum[3] = {0.0, 0.0, 0.0};
            double filtered_value[3] = {0.0, 0.0, 0.0};

            // Get center pixel pointer
            unsigned char *center_pixel = src + (y * width + x) * channels;

            // Iterate over local window
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    int nx = x + j - radius;
                    int ny = y + i - radius;

                    // Bounds check to ensure we're within the image
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }

                    // Get neighbor pixel pointer
                    unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

                    for (int c = 0; c < channels; c++) {
                        // Compute range weight
                        double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
                        double weight = spatial_weights[i * d + j] * range_weight;

                        // Accumulate weighted sum
                        filtered_value[c] += neighbor_pixel[c] * weight;
                        weight_sum[c] += weight;
                    }
                }
            }

            // Normalize and store result
            unsigned char *output_pixel = dst + (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6)); // Avoid division by zero
            }
        }
    }

    free(spatial_weights);
}

// Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image!\n");
        return 1;
    }

    // Ensure that image is not too small for bilateral filter (at least radius of d/2 around edges)
    if (width <= 5 || height <= 5) {
        printf("Image is too small for bilateral filter (at least 5x5 size needed).\n");
        stbi_image_free(image);
        return 1;
    }

    // Allocate memory for output image
    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
    if (!filtered_image) {
        printf("Memory allocation for filtered image failed!\n");
        stbi_image_free(image);
        return 1;
    }
    
    // Mesurer le temps avant l'exécution du filtre
    clock_t start_time = clock();

    // Appliquer le filtre bilatéral
    bilateral_filter(image, filtered_image, width, height, channels, 5, 75.0, 75.0);

    // Mesurer le temps après l'exécution du filtre
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;  // Temps en secondes
    printf("%.2f\n", time_taken);

    // Save the output image
    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Error saving the image!\n");
        free(filtered_image);
        stbi_image_free(image);
        return 1;
    }

    // Free memory
    stbi_image_free(image);
    free(filtered_image);

    printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
    printf("Time taken for bilateral filtering: %f seconds\n", time_taken);

    return 0;
}
