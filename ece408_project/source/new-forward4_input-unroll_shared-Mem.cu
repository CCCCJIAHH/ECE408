#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
#define CUDA_MAX_NUM_THREADS 1024

// optimization 4: input unroll and shared memory



__global__ void unroll_kernel(const float* x, float* X_unroll, int b, int C, int H, int W, int K){
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int W_unroll = H_out * W_out;

  if(t < C * W_unroll) {
    int c = t / W_unroll;
    int s = t % W_unroll;
    int h_out = s / W_out;
    int w_out = s % W_out;
    int h_unroll = h_out * W_out + w_out;
    int w_base = c*K*K;
    for(int p = 0; p < K; p++) {
      for(int q = 0; q < K; q++) {
        int w_unroll = w_base + p * K + q;
        X_unroll[w_unroll * W_unroll + h_unroll] = x4d(b, c, h_out+p, w_out+q);
      }
    }
  }
#undef x4d
}

__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];  // must constant parameter
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  
  float cValue = 0;
  for (int q = 0; q < (numAColumns - 1)/TILE_WIDTH + 1; ++q) {
    
    if (row < numARows && ((q*TILE_WIDTH+tx) < numAColumns)) {  // still in A's range
      subTileA[ty][tx] = A[row*numAColumns + q*TILE_WIDTH+tx];
    } else {
      subTileA[ty][tx] = 0;
    }
    if (q*TILE_WIDTH+ty < numBRows && (col < numBColumns)) {  // still in B's range
      subTileB[ty][tx] = B[(q*TILE_WIDTH+ty)*numBColumns + col];
    } else {
      subTileB[ty][tx] = 0;
    }
    __syncthreads();
        
    if (row < numCRows && col < numCColumns){  // in C's range
      for (int k=0; k<TILE_WIDTH; ++k)
        cValue += subTileA[ty][k] * subTileB[k][tx];  // store temp_value in cValue, need to consider halo cells
    }
    __syncthreads();
  }
  
  if ((row < numCRows) && (col<numCColumns)) {
    C[row*numCColumns + col] = cValue;
  }
  
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
  // Allocate memory and copy over the relevant data structures to the GPU
  const int H_out = H - K + 1;
  const int W_out = W - K + 1; 
  int input_length = B * C * H * W;
  int output_length = B * M * H_out * W_out;
  int kernel_length = M * C * K * K;
  cudaMalloc((void **) device_y_ptr, output_length * sizeof(float));
  cudaMalloc((void **) device_x_ptr, input_length * sizeof(float));
  cudaMalloc((void **) device_k_ptr, kernel_length * sizeof(float));

  // We pass [double pointers] for you to initialize the relevant device pointers,
  //  which are passed to the other two functions.
  cudaMemcpy(*device_x_ptr, host_x, input_length * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(*device_k_ptr, host_k, kernel_length * sizeof(float), cudaMemcpyHostToDevice);

  // // Useful snippet for error checking
  // cudaError_t error = cudaGetLastError();
  // if(error != cudaSuccess)
  // {
  //   std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
  //   exit(-1);
  // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
  // Set the kernel dimensions and call the kernel
  // chapter16, P20
  const int W_out = W - K + 1;
  const int H_out = H - K + 1;
  int W_unroll = H_out * W_out;
  int H_unroll = C * K * K;
  float* X_unrolled;
  cudaMalloc((void **) &X_unrolled, W_unroll * H_unroll * sizeof(float));
  // block num for unroll_kernel
  int num_blocks = ceil(1.0 * C * H_out * W_out / CUDA_MAX_NUM_THREADS);

  dim3 dimGrid(ceil(1.0 * W_unroll / TILE_WIDTH), ceil(1.0 * M / TILE_WIDTH), 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  for(int n = 0; n < B; n++) {
    unroll_kernel<<<num_blocks, CUDA_MAX_NUM_THREADS>>>(device_x, X_unrolled, n, C, H, W, K);
    cudaDeviceSynchronize();
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(device_k, X_unrolled, &device_y[n * M * H_out * W_out], 
                                                M,H_unroll, H_unroll, W_unroll, M, W_unroll);
    cudaDeviceSynchronize();
  }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
  // Copy the output back to host
  const int H_out = H - K + 1;
  const int W_out = W - K + 1; 
  int output_length = B * M * H_out * W_out;
  cudaMemcpy(host_y, device_y, output_length * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(device_y);
  cudaFree(device_x);
  cudaFree(device_k);

}


__host__ void GPUInterface::get_device_properties()
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for(int dev = 0; dev < deviceCount; dev++)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
    std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
    std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
    std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
    std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
    std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
    std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
    std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
    std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
  }
}
