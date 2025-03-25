#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// Fonction Gaussienne sur GPU
__device__ double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// CUDA kernel optimisé (calcul des poids spatiaux et application du filtre)
__global__ void bilateral_filter_kernel(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Récupération de l'abscisse (x) du pixel dans l'image 2D
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Récupération de l'ordonnée (y) du pixel dans l'image 2D

    // Eviter les bords de l'image
    /*
        for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
    */
    int radius = d / 2;
    if (x >= radius && x < width - radius && y >= radius && y < height - radius) {
        double weight_sum[3] = {0.0, 0.0, 0.0};
        double filtered_value[3] = {0.0, 0.0, 0.0};

        unsigned char *center_pixel = src + (y * width + x) * channels;

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int nx = x + j;
                int ny = y + i;

                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

                // Calcul de spatial_weights, le calcul est répété pour des pixels, autre solution est de calculer avant dans un autre kernel puis de le recalculer ici
                double spatial_weight = gaussian(sqrtf(i * i + j * j), sigma_space);

                for (int c = 0; c < channels; c++) {
                    double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
                    double weight = spatial_weight * range_weight;

                    filtered_value[c] += neighbor_pixel[c] * weight;
                    weight_sum[c] += weight;
                }
            }
        }

        unsigned char *output_pixel = dst + (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6)); // Avoid division by zero
        }
    }
}


// Fonction pour appliquer le filtre bilatéral avec CUDA
void bilateral_filter(unsigned char *src, unsigned char *h_dst, int width, int height, int channels, int d, double sigma_color, double sigma_space, int block_size) {
    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_dst, width * height * channels * sizeof(unsigned char));
    cudaMemcpy(d_src, src, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(block_size, block_size); // taille des blocs, 32 max car l'on peut exécuter que 1024 threads par blocs
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // taille de la grille, blockSize*gridSize = 512 pour l'image 
    //printf("Block Size: (%d, %d)\n", blockSize.x, blockSize.y);
    //printf("Grid Size: (%d, %d)\n", gridSize.x, gridSize.y);
    //int maxThreadsPerBlock;
    //cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    //printf("Max Threads per block: %d\n", maxThreadsPerBlock);

    bilateral_filter_kernel<<<gridSize, blockSize>>>(d_src, d_dst, width, height, channels, d, sigma_color, sigma_space);
    // attente du noyau
    cudaDeviceSynchronize();

    cudaMemcpy(h_dst, d_dst, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
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

    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
    if (!filtered_image) {
        printf("Memory allocation for filtered image failed!\n");
        stbi_image_free(image);
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    bilateral_filter(image, filtered_image, width, height, channels, 5, 75.0, 75.0, block_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("%.2f\n", duration.count());

    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Error saving the image!\n");
        free(filtered_image);
        stbi_image_free(image);
        return 1;
    }

    stbi_image_free(image);
    free(filtered_image);
    printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
    return 0;
}
