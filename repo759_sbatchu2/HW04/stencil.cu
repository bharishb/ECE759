#include <cuda.h>
#include "stencil.cuh"
#include <iostream>
using namespace std;

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){


	//shared memory
	extern __shared__ float s[];
	float* shared_mask = s;   // 2*R+1 size

	//Let say thread block starts at index x1 and ends at x2. image values needed are from x1-R to x2+R. That is x2-x1+2R+1 size. That is blockDim.x+2R size
        float* image_thread_block = (float*)&shared_mask[2*R+1]; //threads_per_block + 2*R size
	float* output_thread_block = (float*)&image_thread_block[blockDim.x + 2*R]; //threads_per_block

	if(threadIdx.x < (2*R +1)) {
	    shared_mask[threadIdx.x] = mask[threadIdx.x];
	}

	
        int index1 = blockDim.x*blockIdx.x - R;  // x1 = blockDim1.x*blockIdx.x is the starting index
	for(int k =0; (k*blockDim.x+threadIdx.x)<(blockDim.x + 2*R); k++) {  // Some threads of the block may need to pull more than 1 data per thread. This is loop based of thread index pulling more than 1 data per thread.
		    int i = k*blockDim.x+threadIdx.x;
		    if(((index1 + i)<0) || ((index1 + i)>(n-1))) {
		    	image_thread_block[i] = 1;
		    } else {
		    	image_thread_block[i] = image[index1+i];  //x1 - R to x2 + R : x1, x2 are start and end indices of thread block
		    }
	}
       
        	

        __syncthreads();
	
	if((blockDim.x*blockIdx.x + threadIdx.x)<n) {
	//output
	float output_conv = 0.0; // trying to write into local thread register than memory
        for(int j=0; j<(2*R+1) ; j++)
	{
	    output_conv += image_thread_block[j + threadIdx.x]*shared_mask[j];
	}
	output_thread_block[threadIdx.x] = output_conv;
	}

	__syncthreads();

	//writing back to global memory
	if((blockDim.x*blockIdx.x + threadIdx.x)<n) {
        output[blockDim.x*blockIdx.x + threadIdx.x] = output_thread_block[threadIdx.x];
	}
}


__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block){


      //kernel call
      stencil_kernel<<<(n+threads_per_block-1)/threads_per_block,threads_per_block,(2*R+1)*sizeof(float)+(2*R+threads_per_block)*sizeof(float)+(threads_per_block)*sizeof(float)>>>(image,mask, output, n, R);

}
