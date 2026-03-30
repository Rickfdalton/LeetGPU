#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
        unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

        if(col < K && row < M){
            float sum = 0.0;
            for(unsigned int i=0; i< N; i++){
                sum+= A[row*N + i] * B[i*K + col];
            }
            C[row*K + col]= sum;
        }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //allocate GPU memory
    float* A_d;
    float* B_d;
    float* C_d;

    cudaMalloc((void**)&A_d, M*N*sizeof(float));
    cudaMalloc((void**)&B_d, N*K*sizeof(float));
    cudaMalloc((void**)&C_d, M*K*sizeof(float));

    //copy data to GPU
    cudaMemcpy(A_d,A,M*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,N*K*sizeof(float),cudaMemcpyHostToDevice);

    //call kernel
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, M, N, K);
    cudaDeviceSynchronize();

    //copy data from GPU
    cudaMemcpy(C,C_d,M*K*sizeof(float),cudaMemcpyDeviceToHost);

    //free data
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
}
