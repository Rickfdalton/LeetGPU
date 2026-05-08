#include <cuda_runtime.h>
#define BLOCK_DIM 1024
#define COARSE_FACTOR 96

__global__ void reduction_kernel(const float* input1_d, const float* input2_d, float* partial_sums,int N){
    __shared__ float input_block_s[BLOCK_DIM];

    int id =  (blockDim.x * 2 * COARSE_FACTOR) * blockIdx.x  + threadIdx.x;

    float sum_intial =0.0f;
    for(int i=0; i<COARSE_FACTOR*2; i++){
    
        if(i*BLOCK_DIM + id< N) {
            float val1 = input1_d[i*BLOCK_DIM + id];
            float val2 = input2_d[i*BLOCK_DIM + id];
            float val = (val1-val2)*(val1-val2);
            sum_intial+= val;
        }
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


// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    int threadsPerBlock = BLOCK_DIM;
    int elementsPerBlock = 2*threadsPerBlock* COARSE_FACTOR;
    int blocksPerGrid = (N+elementsPerBlock -1)/elementsPerBlock;
    float* partial_sums= (float*)malloc(blocksPerGrid*sizeof(float));
    float* partial_sums_d;
    cudaMalloc((void**)&partial_sums_d, blocksPerGrid*sizeof(float));
    cudaMemset(partial_sums_d, 0, blocksPerGrid * sizeof(float));

    reduction_kernel<<<blocksPerGrid,threadsPerBlock>>>(predictions, targets,partial_sums_d,N);
    cudaMemcpy(partial_sums,partial_sums_d,blocksPerGrid*sizeof(float),cudaMemcpyDeviceToHost);

    float val=0;
    for(int i =0 ; i<blocksPerGrid; i++){
        val+= partial_sums[i];
    }
    val= val/float(N);
    cudaMemcpy(mse, &val, sizeof(float), cudaMemcpyHostToDevice);
    free(partial_sums);
    cudaFree(partial_sums_d);

}
