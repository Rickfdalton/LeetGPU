#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* matrix_d, float* transpose_d, int rows, int cols) {
    unsigned int row_trans = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col_trans = blockDim.y * blockIdx.y + threadIdx.y;

    if(col_trans < rows && row_trans < cols){
        transpose_d[row_trans*rows+col_trans] = matrix_d[col_trans*cols + row_trans];
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();


}
