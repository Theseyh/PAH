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

// CUDA kernel pour calculer les poids spatiaux
__global__ void calculate_spatial_weights(double *spatial_weights, int d, double sigma_space) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < d && y < d) {
        int radius = d / 2;
        int dx = x - radius;
        int dy = y - radius;
        int idx = y * d + x;
        spatial_weights[idx] = gaussian(sqrtf(dx * dx + dy * dy), sigma_space);
    }
}

// CUDA kernel pour appliquer le filtre bilatéral
__global__ void bilateral_filter_kernel(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double *spatial_weights) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = d / 2;
    if (x >= radius && x < width - radius && y >= radius && y < height - radius) {
        double weight_sum[3] = {0.0, 0.0, 0.0};
        double filtered_value[3] = {0.0, 0.0, 0.0};

        unsigned char *center_pixel = src + (y * width + x) * channels;

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                int nx = x + j - radius;
                int ny = y + i - radius;

                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;
                int idx = (i * d) + j; // Index des poids spatiaux

                for (int c = 0; c < channels; c++) {
                    double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
                    double weight = spatial_weights[idx] * range_weight;

                    filtered_value[c] += neighbor_pixel[c] * weight;
                    weight_sum[c] += weight;
                }
            }
        }

        unsigned char *output_pixel = dst + (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6));
        }
    }
}

// Fonction pour appliquer le filtre bilatéral avec CUDA
void bilateral_filter(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space, int block_size) {
    int radius = d / 2;
    double *d_spatial_weights;
    cudaMalloc(&d_spatial_weights, d * d * sizeof(double));

    dim3 blockSize2D(block_size, block_size);
    dim3 gridSize((d + blockSize2D.x - 1) / blockSize2D.x, (d + blockSize2D.y - 1) / blockSize2D.y);

    calculate_spatial_weights<<<gridSize, blockSize2D>>>(d_spatial_weights, d, sigma_space);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in calculate_spatial_weights: %s\n", cudaGetErrorString(err));
        return;
    }

    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_dst, width * height * channels * sizeof(unsigned char));
    cudaMemcpy(d_src, src, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    gridSize = dim3((width + blockSize2D.x - 1) / blockSize2D.x, (height + blockSize2D.y - 1) / blockSize2D.y);

    bilateral_filter_kernel<<<gridSize, blockSize2D>>>(d_src, d_dst, width, height, channels, d, sigma_color, d_spatial_weights);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in bilateral_filter_kernel: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(dst, d_dst, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_spatial_weights);
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
