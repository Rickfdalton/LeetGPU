#include <cuda_runtime.h>
#include <math.h>
#define BLOCK_SIZE 1024
#define COARSE_FACTOR 4


__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


__global__ void soft_max_kernel(const float* input, float* denom, float* max_value, float* output, int N){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float max_val = *max_value;
    float denominator= *denom;

    if(index<N){
        output[index] = expf(input[index]-max_val)/denominator ;
    }
}

__global__ void max_reduce_kernel(const float* input, float* max_val,int N) {
    int index = blockDim.x *blockIdx.x * 2* COARSE_FACTOR + threadIdx.x;

    __shared__ float block_max_s[BLOCK_SIZE];

    float max_partial= -INFINITY;
    for(int i=0; i< 2*COARSE_FACTOR; i++){
        if(index + i*BLOCK_SIZE < N){
            max_partial=fmaxf(max_partial,input[index + i*BLOCK_SIZE]);
        }
    }
    block_max_s[threadIdx.x] = max_partial;

    __syncthreads();

    for(int stride=BLOCK_SIZE/2; stride>0; stride/=2){
        if(threadIdx.x < stride){
            block_max_s[threadIdx.x] = fmaxf(block_max_s[threadIdx.x],block_max_s[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if(threadIdx.x==0){
        atomicMax(max_val,block_max_s[0]);
    }
}


__global__ void denominator_kernel(const float* input, float* denominator, float* max_val,int N) {
    int index = blockDim.x *blockIdx.x * 2* COARSE_FACTOR + threadIdx.x;

    __shared__ float block_s[BLOCK_SIZE];

    float sum_initial=0.0f;
    for(int i=0; i< 2*COARSE_FACTOR; i++){
        if(index + i*BLOCK_SIZE < N){
            sum_initial+= expf(input[index + i*BLOCK_SIZE]- *max_val);
        }
    }
    block_s[threadIdx.x] = sum_initial;

    __syncthreads();

    for(int stride=BLOCK_SIZE/2; stride>0; stride/=2){
        if(threadIdx.x < stride){
            block_s[threadIdx.x]+=block_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x==0){
        atomicAdd(denominator,block_s[0]);
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCK_SIZE;
    int elementsPerBlock = threadsPerBlock*2*COARSE_FACTOR;
    int blocksPerGrid = (N + elementsPerBlock - 1) / elementsPerBlock;

    float* denominator;
    float* max;
    cudaMalloc((void**)&denominator, sizeof(float)); 
    cudaMalloc((void**)&max, sizeof(float));   
    cudaMemset(denominator,0,sizeof(float)); 
    float neg_inf =  -INFINITY;
    cudaMemcpy(max, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    max_reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, max, N);
    cudaDeviceSynchronize();

    denominator_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, denominator, max, N);
    cudaDeviceSynchronize();
    soft_max_kernel<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(input, denominator,max, output, N);

    cudaFree(denominator);
    cudaFree(max);

}
