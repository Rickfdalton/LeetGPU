#include <cuda_runtime.h>
#define BLOCKSIZE 32

__global__ void convolution_kernel(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols){
        int col = blockDim.x*blockIdx.x + threadIdx.x ;
        int row = blockDim.y*blockIdx.y + threadIdx.y ;
        
        if(col <= input_cols-kernel_cols && row <= input_rows-kernel_rows ){
            float sum=0.0;
            for(int mask_row=0; mask_row< kernel_rows; mask_row++){
                for(int mask_col=0; mask_col< kernel_cols; mask_col++){
                    sum+=kernel[mask_row*kernel_cols + mask_col] * input[(row+mask_row) * input_cols + col + mask_col];
                }
            }
            output[(row*input_cols + col)-(kernel_cols-1)*row]= sum; 
        }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    
    dim3 threadsPerBlock(BLOCKSIZE,BLOCKSIZE,1);
    dim3 blocksPerGrid((input_cols+BLOCKSIZE -1)/BLOCKSIZE,(input_rows+BLOCKSIZE -1)/BLOCKSIZE,1);

    convolution_kernel<<< blocksPerGrid , threadsPerBlock >>>(input,kernel,output,input_rows,input_cols,kernel_rows,kernel_cols);
    cudaDeviceSynchronize();

}


/*
I have written non tiled code, can further optimize with tiling
*/