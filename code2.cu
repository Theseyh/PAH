#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

__device__ double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

__global__ void bilateral_filter_kernel(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int radius = d / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    double weight_sum[3] = {0.0, 0.0, 0.0};
    double filtered_value[3] = {0.0, 0.0, 0.0};
    unsigned char *center_pixel = src + (y * width + x) * channels;

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int nx = x + j;
            int ny = y + i;

            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;
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
        output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6));
    }
}

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

    size_t img_size = width * height * channels * sizeof(unsigned char);
    unsigned char *filtered_image = (unsigned char *)malloc(img_size);
    if (!filtered_image) {
        printf("Memory allocation failed!\n");
        stbi_image_free(image);
        return 1;
    }

    unsigned char *d_src, *d_dst;
    cudaMalloc((void **)&d_src, img_size);
    cudaMalloc((void **)&d_dst, img_size);
    cudaMemcpy(d_src, image, img_size, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    bilateral_filter_kernel<<<grid_size, block_size>>>(d_src, d_dst, width, height, channels, 5, 75.0, 75.0);
    cudaDeviceSynchronize();

    cudaMemcpy(filtered_image, d_dst, img_size, cudaMemcpyDeviceToHost);

    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Error saving the image!\n");
    }

    stbi_image_free(image);
    free(filtered_image);
    cudaFree(d_src);
    cudaFree(d_dst);

    printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
    return 0;
}
