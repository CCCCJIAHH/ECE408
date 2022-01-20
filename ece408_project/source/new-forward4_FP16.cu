#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"
#define TILE_WIDTH 16 //fixed for tensors
#define KERNEL_SIZE 7

__constant__ float K_ONE[4*KERNEL_SIZE*KERNEL_SIZE];
__constant__ float K_TWO[16*4*KERNEL_SIZE*KERNEL_SIZE];

//C = A*B
__global__ void matrixMultiplyShared( const float *x,  const float *k, float *y,
                                     const int B, const int M, const int C, const int H, const int W, const int K) {

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int x_rows = K*K*C;
    const int x_cols = H_out*W_out;
    const int k_rows = M;
    const int k_cols = K*K*C;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k4d_1(i3, i2, i1, i0) K_ONE[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k4d_2(i3, i2, i1, i0) K_ONE[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]



    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ __half rowTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half colTile[TILE_WIDTH][TILE_WIDTH];


    for(int b=0; b<B; b++){
        __half accum = 0;



        for(int p=0; p<ceil(k_cols / (float)TILE_WIDTH); ++p){

            int c = (threadIdx.x + p*TILE_WIDTH)/(K*K);
            int k_remain =  (threadIdx.x + p*TILE_WIDTH) % (K*K);
            if(M==4){

                if (row < k_rows && (threadIdx.x + p*TILE_WIDTH) < k_cols)
                    rowTile[threadIdx.y][threadIdx.x] = __float2half(k4d(row, c, k_remain/K, k_remain%K));
                else
                    rowTile[threadIdx.y][threadIdx.x] = 0.0;
            }
            else{
                if (row < k_rows && (threadIdx.x + p*TILE_WIDTH) < k_cols)
                    rowTile[threadIdx.y][threadIdx.x] = __float2half(k4d(row, c, k_remain/K, k_remain%K));
                else
                    rowTile[threadIdx.y][threadIdx.x] = 0.0;
            }

            c = (threadIdx.y + p*TILE_WIDTH)/(K*K);
            int x_remain =  (threadIdx.y + p*TILE_WIDTH) % (K*K);
            int y_in = col/W_out;
            int x_in = col%W_out;

            if (col < x_cols && (threadIdx.y+TILE_WIDTH*p) < x_rows)
                //colTile[threadIdx.y][threadIdx.x] = x_unroll_4d(b, (threadIdx.y + p*TILE_WIDTH), col);
                colTile[threadIdx.y][threadIdx.x] = __float2half(x4d(b,c, y_in + x_remain/K, x_in + x_remain%K));
            else
                colTile[threadIdx.y][threadIdx.x] = 0.0;

            __syncthreads();

            for(int i=0; i<TILE_WIDTH; ++i){
                accum += rowTile[threadIdx.y][i] * colTile[i][threadIdx.x];
            }
            __syncthreads();

        }
        if (row < M && col < H_out*W_out)
            y4d(b, row, col/W_out, col%W_out) = __half2float(accum);
    }

}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k,
                                                    float **device_y_ptr, float **device_x_ptr, float **device_k_ptr,
                                                    const int B, const int M, const int C, const int H, const int W,
                                                    const int K) {


    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    cudaMalloc(device_y_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc(device_x_ptr, B * C * H * W * sizeof(float));
    cudaMalloc(device_k_ptr, M * C * K * K * sizeof(float));

    bool val;
    val=cudaMemcpy(*device_x_ptr, host_x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
   val= cudaMemcpy(*device_k_ptr, host_k, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);



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

    int B_interval = (B>5000)?10000:B;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    if (M == 4)	cudaMemcpyToSymbol(K_ONE, device_k, M*C*K*K*sizeof(float));
    if (M == 16) cudaMemcpyToSymbol(K_TWO, device_k, M*C*K*K*sizeof(float));

    for (int b = 0; b<B; b+=B_interval){
        //val = cudaMemcpy(device_x, host_x + b*C*H*W, (B_interval*C*H*W) * sizeof(float), cudaMemcpyHostToDevice);

        dim3 gridDim(ceil(H_out*W_out/(1.0 * TILE_WIDTH)), ceil(M/(1.0 * TILE_WIDTH)), 1);
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        matrixMultiplyShared<<< gridDim, blockDim>>>(device_x, device_k, device_y,
                                                     B_interval, M, C, H, W, K);
        //val = cudaMemcpy(host_y + b*M*H_out*W_out, device_y, (B_interval*M*H_out*W_out) * sizeof(float), cudaMemcpyDeviceToHost);
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
