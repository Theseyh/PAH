#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>


__device__ double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}



__global__ void bilateral_filter_kernel(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int radius = d / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /*on fait les conditions aux limites*/
    if (x >= radius && x < width - radius && y >= radius && y < height - radius) {
        double weight_sum[3] = {0.0, 0.0, 0.0};
        double filtered_value[3] = {0.0, 0.0, 0.0};

        // Get center pixel pointer
        unsigned char *center_pixel = src + (y * width + x) * channels;

        // Precompute spatial Gaussian weights for the current pixel
        double spatial_weights[25]; // Maximum window size of 5x5
        
        int count = 0;
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                spatial_weights[count++] = gaussian(sqrtf(i * i + j * j), sigma_space);

            }
        }

        // Iterate over local window
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int nx = x + j;
                int ny = y + i;

                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                // Get neighbor pixel pointer
                unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

                for (int c = 0; c < channels; c++) {
                    // Compute range weight
                    double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
                    double weight = spatial_weights[(i + radius) * d + (j + radius)] * range_weight;
                    

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



// Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image> [block_size]\n", argv[0]);
        return 1;
    }

    int block_size = 16;
    if (argc == 4) {
        block_size = atoi(argv[3]);
        if (block_size <= 0) {
            printf("Invalid block size! Using default block size of 16.\n");
            block_size = 16;
        }
    }

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image!\n");
        return 1;
    }

    if (width <= 5 || height <= 5) {
        printf("Image is too small for bilateral filter (at least 5x5 size needed).\n");
        stbi_image_free(image);
        return 1;
    }

    unsigned char *d_src, *d_dst;
    size_t img_size = width * height * channels;
    cudaMalloc((void **)&d_src, img_size);
    cudaMalloc((void **)&d_dst, img_size);
    cudaMemcpy(d_src, image, img_size, cudaMemcpyHostToDevice);

    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);

    auto start = std::chrono::high_resolution_clock::now();
    bilateral_filter_kernel<<<grid, block>>>(d_src, d_dst, width, height, channels, 5, 75.0, 75.0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    unsigned char *filtered_image = (unsigned char *)malloc(img_size);
    cudaMemcpy(filtered_image, d_dst, img_size, cudaMemcpyDeviceToHost);

    std::chrono::duration<double> duration = end - start;
    printf("Temps total (CPU + GPU) : %.3f secondes\n", duration.count());

    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Error saving the image!\n");
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    stbi_image_free(image);
    free(filtered_image);
    printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
    return 0;
}