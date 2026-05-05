#include <cuda_runtime.h>

__device__ void merge_seq(const float* A, const float* B, float* C,   int m,   int n);
__device__   int coRank(const float* A, const float* B,   int m,   int n,   int k);
__global__ void merge_kernel(const float* A, const float* B, float* C,   int m,   int n);

/* merging 2 sorted array */
__device__ void merge_seq(const float* A, const float* B, float* C,   int m,   int n){
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
__device__   int coRank(const float* A, const float* B,   int m,   int n,   int k){
      int low = max(k-n,0);
      int high = min(k,m);

    while (true){
          int i = (low+high)/2;
          int j = k-i;
        if(i>0 && j<n && A[i-1]>B[j]){
            high=i;
        }else if(j>0 && i<m && B[j-1]>A[i]){
            low=i+1;
        }else{
            return i;
        }
    }
}

/*
merge kernel
each thread is responsible for calculating element for each index of output array
*/
#define ELEM_PER_THREAD 6
#define THREADS_PER_BLOCK 128
#define ELEM_PER_BLOCK (ELEM_PER_THREAD*THREADS_PER_BLOCK)

__global__ void merge_kernel(const float* A, const float* B, float* C,   int m,   int n){
    //find block's segments
    int kBlock =  blockIdx.x* ELEM_PER_BLOCK;
    int kNextBlock = (blockIdx.x < gridDim.x -1) ? kBlock+ELEM_PER_BLOCK: m+n;

    __shared__ int iBlock;
    __shared__ int iNextBlock;

    if(threadIdx.x == 0){
        iBlock = coRank(A,B,m,n,kBlock);
        iNextBlock = coRank(A,B,m,n,kNextBlock);
    }
    __syncthreads();
    int jBlock = kBlock-iBlock;
    int jNextBlock = kNextBlock - iNextBlock;

    __shared__ float A_s[ELEM_PER_BLOCK];
    __shared__ float B_s[ELEM_PER_BLOCK];

    int mBlock = iNextBlock - iBlock;
    int nBlock = jNextBlock - jBlock;

    for(int i= threadIdx.x; i<mBlock; i+=blockDim.x){
        A_s[i]=A[iBlock+i];
    }

    for(int j= threadIdx.x; j<nBlock; j+=blockDim.x){
        B_s[j]=B[jBlock+j];
    }
    __syncthreads();

    // merge
    __shared__ float C_s[ELEM_PER_BLOCK];
    int k = threadIdx.x* ELEM_PER_THREAD;

    if(k < mBlock + nBlock){
          int i = coRank(A_s, B_s, mBlock,nBlock,k);
          int j = k - i;
          int kNext = (k + ELEM_PER_THREAD ) < mBlock+nBlock ? k+ ELEM_PER_THREAD : mBlock+nBlock ;
          int iNext = coRank(A_s,B_s,mBlock,nBlock,kNext);
          int jNext = kNext - iNext;
        merge_seq(&A_s[i], &B_s[j], &C_s[k],iNext-i,jNext-j);
    }
    __syncthreads();

    //write to global memory
    for(int k= threadIdx.x; k<mBlock+nBlock; k+=blockDim.x){
        C[kBlock+k]=C_s[k];
    }
}



// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {

      int blocks_per_grid = (M+N+ELEM_PER_BLOCK -1)/ELEM_PER_BLOCK ;
    merge_kernel <<<blocks_per_grid, THREADS_PER_BLOCK>>> (A, B, C, M,N);

}
