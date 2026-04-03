#include <cuda_runtime.h>
#define BLOCK_DIM 16
#define C_1 0.25


__global__ void stencil_kernel(const float* input, float* output, int rows, int cols){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x==0 || y==0 || x==cols-1 || y==rows-1){
        output[cols*y + x]= input[cols*y + x];
    }
    
    if(x>0 && x<cols-1 && y>0 && y<rows-1){
        output[cols*y + x] = 
                    C_1 *
                    (
                    input[cols*(y-1) + x]+input[cols*(y+1) + x]+
                    input[cols*y + (x-1)]+input[cols*y + x+1]
                    );
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_DIM,BLOCK_DIM,1);
    dim3 blocksPerGrid((cols+BLOCK_DIM-1)/BLOCK_DIM, (rows+BLOCK_DIM-1)/BLOCK_DIM,1);


    stencil_kernel<<<blocksPerGrid,threadsPerBlock>>>(input,output,rows,cols);
}
