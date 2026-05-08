#include <iostream>
#include <cuda_runtime.h>

__device__   int coRank(float* A, float* B,   int m,   int n,   int k);
__device__ void merge(float* A, float* B, float* C,   int m,   int n);
__global__ void merge_sort_kernel(float* src, float* dst, int N, int width);

/*
Finding i from k
ie. when the index of the output array given find the correct index of input array A.
from that we can find the index of the input array B.
*/
__device__   int coRank(float* A, float* B,   int m,   int n,   int k){
      int low = max(k-n,0);
      int high = min(k,m);

    while (low < high){
          int i = (low+high)/2;
          int j = k-i;
        if(i>0 && j<n && A[i-1]>B[j]){
            high=i;
        }else if(j>0 && i<m && B[j-1]>A[i]){
            low=i+1;
        }else{
            return i;
        }
    }return low;
}

/*
merge kernel
each thread is responsible for calculating element for each index of output array
*/
#define BLOCK_SIZE 512


__device__ void merge(float* A, float* B, float* C,   int m,   int n){  
    // merge
    for (int k = threadIdx.x; k < m + n; k += blockDim.x){
          int i = coRank(A, B, m,n,k);
          int j = k - i;
           if (i < m && (j >= n || A[i] <= B[j])) {
                 C[k] = A[i];
            } else {
                 C[k] = B[j];
            }
    }
    __syncthreads();
}

__global__ void merge_sort_kernel(float* src, float* dst, int N, int width){
    int left = blockIdx.x * 2 * width;
    int right = left + width;

    if (right < N){
        int m = min(width, N-left);
        int n = min(width, N-right);
        merge(
            &src[left],
            &src[right],
            &dst[left],
            m,
            n
        );
    }else{
        for(int i = threadIdx.x; i < min(width, N-left); i += blockDim.x)
            dst[left + i] = src[left + i];
        return;
    }
}

// data is device pointer
extern "C" void solve(float* data, int N) {
    float* A_d;
    float* C_d;

    cudaMalloc((void**) &A_d, N*sizeof(float));
    cudaMalloc((void**) &C_d, N*sizeof(float));

    cudaMemcpy(A_d, data, N*sizeof(float), cudaMemcpyHostToDevice);

    for(int stride=1; stride< N; stride*=2){
    int blocks_per_grid = (N+2*stride -1)/(2*stride) ;
        merge_sort_kernel <<<blocks_per_grid, BLOCK_SIZE >>> (A_d, C_d, N, stride);
        float *temp = C_d;
        C_d = A_d;
        A_d = temp;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(data, A_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(C_d);
}
