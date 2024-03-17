#include <cuda.h>
#include "reduce.cuh"
#include <stdio.h>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ float sdata[];
    unsigned int index = threadIdx.x + 2*blockDim.x*blockIdx.x;
    //printf("thread index : %d , blockIdx : %d, blockDim : %d, n : %d, index : %d, value : %f, input1 : %f input2 : %f\n", threadIdx.x , blockIdx.x, blockDim.x, n, index, sdata[threadIdx.x], g_idata[index],g_idata[index + blockDim.x]);
    if(index < n) {
        if((index + blockDim.x)<n){
            sdata[threadIdx.x] = g_idata[index] + g_idata[index + blockDim.x];
        } else{
            sdata[threadIdx.x] = g_idata[index];
        }
    } else {
        sdata[threadIdx.x] = 0.0;
    }
    __syncthreads(); //making sure data is available in shared memory

    //shared mem print
    //printf("thread index : %d , blockIdx : %d , value : %f, input1 : %f input2 : %f\n", threadIdx.x , blockIdx.x, sdata[threadIdx.x], g_idata[index],g_idata[index + blockDim.x]);

    //reduce in shared memory - Tried handling non 2 power block sizes
    for(unsigned int s=blockDim.x; s>1; s=(s+1)/2){
        if(threadIdx.x < (s)/2){
            sdata[threadIdx.x] += sdata[threadIdx.x+(s+1)/2];
        }
        __syncthreads();
    }
    g_odata[blockIdx.x] = sdata[0];
}


__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block)
{
   //launch multiple times to ensure 1 thread block left for execution
   unsigned int i;
   for(i = N; i>2*threads_per_block; i=(i+2*threads_per_block-1)/(2*threads_per_block)){
    //printf("host function calling kernel i : %d\n", i);
       reduce_kernel<<<(i+2*threads_per_block-1)/(2*threads_per_block), threads_per_block, threads_per_block*sizeof(float)>>>(*input, *output, i);        
   }

    reduce_kernel<<<1, threads_per_block, threads_per_block*sizeof(float)>>>(*input, *output, i);        

}