#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if(col < width && row < height){
        int idx = row * width + col;
        int red_idx = idx*3;
        int green_idx = red_idx+ 1;
        int blue_idx = green_idx+ 1;

        output[idx]= 0.299* input[red_idx] + 0.587*input[green_idx] + 0.114*input[blue_idx];
        
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    dim3 threadsPerBlock(16,16,1);
    dim3 blocksPerGrid((width + 16 - 1) / 16,(height + 16 - 1) / 16 );

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
