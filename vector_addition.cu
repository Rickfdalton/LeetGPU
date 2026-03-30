#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* A_d;
    float* B_d;
    float* C_d;

    cudaMalloc((void**)&A_d, N*sizeof(float));
    cudaMalloc((void**)&B_d, N*sizeof(float));
    cudaMalloc((void**)&C_d, N*sizeof(float));

    //copy to GPU
    cudaMemcpy(A_d, A,N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B,N*sizeof(float), cudaMemcpyHostToDevice);

    //kernel
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();

    //copy back
    cudaMemcpy(C, C_d,N*sizeof(float), cudaMemcpyDeviceToHost);

    //free
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}
