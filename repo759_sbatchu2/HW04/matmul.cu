#include "matmul.cuh"
#include <cuda.h>


__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
        
	//compute matrix output element

	if(index < n*n) {
		float sum = 0.0;
		for(size_t k=0; k <n; k++)
		{
		    sum = sum + A[index - index%n + k]*B[k*n + (index%n)];  // doing reduction on registers rather than memory
		}
		C[index] = sum; //writing final value in memory
	}
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block)
{

	//call kernel
	matmul_kernel<<<(n*n+threads_per_block-1)/threads_per_block, threads_per_block>>>(A, B, C, n);  //kernel launch

}
