#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
  /*
  Modify this function to implement the forward pass described in Chapter 16.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct AND fast.

  Function paramter definitions:
  y - output
  x - input
  k - kernel
  B - batch_size (number of images in x)
  M - number of output feature maps
  C - number of input feature maps
  H - input height dimension
  W - input width dimension
  K - kernel height and width (K x K)
  */

  // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
  // An example use of these macros:
  // float a = y4d(0,0,0,0)
  // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
  __shared__ float W_shared[TILE_WIDTH][TILE_WIDTH];
  __shared__ float X_shared[TILE_WIDTH][TILE_WIDTH];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE_WIDTH + ty;
  int column = blockIdx.x * TILE_WIDTH + tx;
  int unroll_column = C * K * K;

  float acc = 0.0;
  int iterations = ceil(1.0 * unroll_column / TILE_WIDTH);

  for (int i = 0; i < iterations; i++) {
    W_shared[ty][tx] = 0;
    X_shared[ty][tx] = 0;

    int absolute_x = i * TILE_WIDTH + tx;
    int absolute_y = i * TILE_WIDTH + ty;

    int W_m = row;
    int W_c = absolute_x / (K * K);
    int W_h = (absolute_x % (K * K)) / K;
    int W_w = (absolute_x % (K * K)) % K;

    if ((absolute_x < unroll_column) && (row < M)){
      W_shared[ty][tx] = k4d(W_m, W_c, W_h, W_w);
    }
    else{
      W_shared[ty][tx] = 0;
    }

    int X_n = bz;
    int X_c = absolute_y / (K * K);
    int X_p = (absolute_y % (K * K)) / K;
    int X_q = (absolute_y % (K * K)) % K;
    int X_h = column / W_out;
    int X_w = column % W_out;
    if (absolute_y < unroll_column && column < H_out * W_out){
      X_shared[ty][tx] = x4d(X_n, X_c, X_h + X_p, X_w + X_q);
    }
    else{
      X_shared[ty][tx] = 0;
    }
    __syncthreads();

    for (int q = 0; q < TILE_WIDTH; q++){
      acc += W_shared[ty][q] * X_shared[q][tx];
    }
    __syncthreads();
  }

  int Y_n = bz;
  int Y_m = row;
  int Y_h = column / W_out;
  int Y_w = column % W_out;
  if (row < M && column < W_out * H_out)
    y4d(Y_n, Y_m, Y_h, Y_w) = acc;
  
#undef y4d
#undef x4d
#undef k4d

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
  // can we define inside a function, or should we just define outside?

  // why we need to use const here? can we use int directly?
  // chapter16, P13
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  dim3 gridDim(ceil(1.0 * H_out * W_out / TILE_WIDTH), ceil(1.0 * M / TILE_WIDTH), B);
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);

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
