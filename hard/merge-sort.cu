#include <iostream>
#include <cuda_runtime.h>

__device__ void merge_seq(float* A, float* B, float* C,   int m,   int n);
__device__  int coRank(float* A, float* B,   int m,   int n,   int k);
__device__ void merge(float* A, float* B, float* C,   int m,   int n);
__global__ void merge_sort_kernel(float* src, float* dst, int N, int width);

__device__ void merge_seq(float* A, float* B, float* C,   int m,   int n){
      int i=0;
      int j=0;
      int k=0;

    while(i<m and j<n){
        if(A[i]<=B[j]){
            C[k++]=A[i++];
        }else{
            C[k++]=B[j++];
        }
    }
    while(i<m){
        C[k++]=A[i++];
    }
    while(j<n){
        C[k++]=B[j++];
    }
}

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

#define BLOCK_SIZE 256
#define CHUNK 16

__device__ void merge_shared(float* A, float* B, float* C,   int m,   int n){
    extern __shared__ float shared[];
    float* A_s = shared;
    float* B_s = shared + m;
  
    for(int i= threadIdx.x; i<m; i+=blockDim.x){
        A_s[i]=A[i];
    }
    __syncthreads();

    for(int j= threadIdx.x; j<n; j+=blockDim.x){
        B_s[j]=B[j];
    }
    __syncthreads();

    for(int start=threadIdx.x * CHUNK; start<m+n ;start+=blockDim.x* CHUNK){
        int k_start= start;
        int k_end  = min(start+CHUNK -1, m+n-1);

        int i_start = coRank(A_s,B_s,m,n,k_start);
        int j_start = k_start-i_start;

        int i_end = coRank(A_s,B_s,m,n,k_end+1);
        int j_end = k_end-i_end+1;

        merge_seq(&A_s[i_start], &B_s[j_start], &C[k_start], i_end-i_start,   j_end-j_start);
        
    }
   
    __syncthreads();

}

__device__ void merge_global(float* A, float* B, float* C,   int m,   int n){

    for(int start=threadIdx.x * CHUNK; start<m+n ;start+=blockDim.x*CHUNK){
        int k_start= start;
        int k_end  = min(start+CHUNK -1, m+n-1);

        int i_start = coRank(A,B,m,n,k_start);
        int j_start = k_start-i_start;

        int i_end = coRank(A,B,m,n,k_end+1);
        int j_end = k_end-i_end+1;

        merge_seq(&A[i_start], &B[j_start], &C[k_start], i_end-i_start,   j_end-j_start);
        
    }
    
}

__global__ void merge_sort_shared_kernel(float* src, float* dst, int N, int width){
    int left = blockIdx.x * 2 * width;
    int right = left + width;

    if (right < N){
        int m = min(width, N-left);
        int n = min(width, N-right);
        merge_shared(
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

__global__ void merge_sort_kernel(float* src, float* dst, int N, int width){
    int left = blockIdx.x * 2 * width;
    int right = left + width;

    if (right < N){
        int m = min(width, N-left);
        int n = min(width, N-right);
        merge_global(
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
    float* C_d;
    float* A_d;

    cudaMalloc((void**) &A_d, N*sizeof(float));
    cudaMalloc((void**) &C_d, N*sizeof(float));

    cudaMemcpy(A_d, data, N*sizeof(float), cudaMemcpyHostToDevice);

    for(int stride=1; stride< N; stride*=2){
    int blocks_per_grid = (N+2*stride -1)/(2*stride) ;
        if(stride <= 4096){
            merge_sort_shared_kernel <<<blocks_per_grid, BLOCK_SIZE,2*stride*sizeof(float)>>> (A_d, C_d, N, stride);
        }else{
            merge_sort_kernel <<<blocks_per_grid, BLOCK_SIZE>>> (A_d, C_d, N, stride);
        }
        float *temp = C_d;
        C_d = A_d;
        A_d = temp;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(data, A_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(C_d);
}