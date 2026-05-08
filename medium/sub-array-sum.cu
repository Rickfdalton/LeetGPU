#include <cuda_runtime.h>
#define BLOCK_DIM 1024
#define COARSE_FACTOR 96

__global__ void reduction_kernel(const int* input_d, int* partial_sums, int N){
    __shared__ int input_block_s[BLOCK_DIM];

    int id =  (blockDim.x * 2 * COARSE_FACTOR) * blockIdx.x  + threadIdx.x;

    int sum_intial =0;
    for(int i=0; i<COARSE_FACTOR*2; i++){
        if(i*BLOCK_DIM + id< N) sum_intial+= input_d[i*BLOCK_DIM + id];
    }
    input_block_s[threadIdx.x]=sum_intial;
    __syncthreads();
 
    for(int stride=BLOCK_DIM/2; stride>0; stride/=2){
        if (threadIdx.x <stride) {
            input_block_s[threadIdx.x]+= input_block_s[threadIdx.x+stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        partial_sums[blockIdx.x]=input_block_s[0]; 
    }

}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    input = input+ S;
    N = E - S +1 ;

    int threadsPerBlock = BLOCK_DIM;
    int elementsPerBlock = 2*threadsPerBlock* COARSE_FACTOR;
    int blocksPerGrid = (N+elementsPerBlock -1)/elementsPerBlock;
    int* partial_sums= (int*)malloc(blocksPerGrid*sizeof(int));
    int* partial_sums_d;
    cudaMalloc((void**)&partial_sums_d, blocksPerGrid*sizeof(int));
    cudaMemset(partial_sums_d, 0, blocksPerGrid * sizeof(int));

    reduction_kernel<<<blocksPerGrid,threadsPerBlock>>>(input,partial_sums_d,N);
    cudaMemcpy(partial_sums,partial_sums_d,blocksPerGrid*sizeof(int),cudaMemcpyDeviceToHost);

    int sum = 0;
    for(int i =0 ; i<blocksPerGrid; i++){
        sum+= partial_sums[i];
    }
    cudaMemcpy(output, &sum, sizeof(int), cudaMemcpyHostToDevice);
    free(partial_sums);
    cudaFree(partial_sums_d);
}
