#include <stdio.h>
#include <cuda_runtime.h>
//nvcc hello_world.cu -o hello_world
//./hello_world

// CUDA kernel function
__global__ void kernel() {
    printf("Hello from the GPU!\n");
}

int main(void) {
    // Launch the kernel with 1 block and 1 thread
    kernel<<<1, 1>>>();

    // Wait for the GPU to finish before printing from the CPU
    cudaDeviceSynchronize();

    printf("Hello World!\n");
    return 0;
}