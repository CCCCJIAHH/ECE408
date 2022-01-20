#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 1024

__global__ void unroll(const float *x, float *X_unroll, int b, const int C, const int H, const int W, const int K, int H_out, int W_out, int H_unroll, int W_unroll) {
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < C * W_unroll) {
        int c = idx / W_unroll;
        int h_out = (idx % W_unroll) / W_out;
        int w_out = (idx % W_unroll) % W_out;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = c * K * K + p * K + q;
                int w_unroll = h_out * W_out + w_out;
                X_unroll[h_unroll * W_unroll + w_unroll] = x4d(b, c, h_out + p, w_out + q);
            }
        }
    }
#undef x4d
}

__global__ void forward_kernel_unroll(float *X_unroll, float *y, const float *k, int b, const int B, const int M, const int C, const int H, const int W, const int K, int H_out, int W_out, int H_unroll, int W_unroll) {



#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    __shared__ float sharedW[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedX[TILE_WIDTH][TILE_WIDTH];

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float result = 0;

    for (int i = 0; i < (H_unroll + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
        // 1. load W into shared memory
        if (i * TILE_WIDTH + tx < H_unroll && row < M) {
            sharedW[ty][tx] = k[row * H_unroll + i * TILE_WIDTH + tx];
        } else {
            sharedW[ty][tx] = 0.0;
        }

        // 2. load X_unroll into shared memory
        if (i * TILE_WIDTH + ty < H_unroll && col < W_unroll) {
            sharedX[ty][tx] = X_unroll[(i * TILE_WIDTH + ty) * W_unroll + col];
        } else {
            sharedX[ty][tx] = 0.0;
        }

        __syncthreads();

        // 3. matrix multiple
        for (int j = 0; j < TILE_WIDTH; j++) {
            result += (sharedW[ty][j] * sharedX[j][tx]);
        }
        __syncthreads();

        if (row < M && col < W_unroll) {
            y[b * M * W_unroll + row * W_unroll + col] = result;
        }
    }

#undef y4d
#undef x4d
#undef k4d
}
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k,
                                                    float **device_y_ptr, float **device_x_ptr, float **device_k_ptr,
                                                    const int B, const int M, const int C, const int H, const int W,
                                                    const int K) {
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    cudaMalloc(device_y_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc(device_x_ptr, B * C * H * W * sizeof(float));
    cudaMalloc(device_k_ptr, M * C * K * K * sizeof(float));

    cudaMemcpy(*device_x_ptr, host_x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);


    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void
GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M,
                               const int C, const int H, const int W, const int K) {
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int H_grid = ceil((float)H_out/TILE_WIDTH);
    int W_grid = ceil((float)W_out/TILE_WIDTH);

    // Set the unroll kernel
    float *X_unroll;
    int W_unroll = H_out * W_out;
    int H_unroll = C * K * K;
    cudaMalloc((void**) &X_unroll, H_unroll * W_unroll * sizeof(float));

    // Set the kernel dimensions
    dim3 gridDim1(ceil(1.0*(float)C * H_out * W_out / BLOCK_SIZE), 1, 1);
    dim3 blockDim1(BLOCK_SIZE, 1, 1);
    dim3 gridDim2(ceil(1.0*(float)W_unroll / TILE_WIDTH), ceil(1.0*(float)M / TILE_WIDTH), 1);
    dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    for (int b = 0; b < B; b++) {
        unroll<<<gridDim1, blockDim1>>>(device_x, X_unroll, b, C, H, W, K, H_out, W_out, H_unroll, W_unroll);
        cudaDeviceSynchronize();
        forward_kernel_unroll<<<gridDim2, blockDim2>>>(X_unroll, device_y, device_k, b, B, M, C, H, W, K, H_out, W_out, H_unroll, W_unroll);
        cudaDeviceSynchronize();
    }

}


__host__ void
GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B,
                                      const int M, const int C, const int H, const int W, const int K) {
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    cudaMemcpy(host_y, device_y, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
}


__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1]
                  << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1]
                  << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
