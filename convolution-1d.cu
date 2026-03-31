#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    unsigned int output_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(output_idx < input_size - kernel_size + 1){
        float sum=0.0;
        for(int i =0 ; i<kernel_size; i++){
           sum+=kernel[i]*input[output_idx + i];
        }
        output[output_idx]=sum;
    } 
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
