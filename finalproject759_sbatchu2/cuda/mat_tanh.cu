#include <math.h>
#include <cuda.h>
#include "LSTM.cuh"

template<typename T>
__global__ void mat_tanh_kernel(const T* A, T* C, int p, int q) {

        //Addition by reading from Global Memory

printf("kernel call tanh: %d, blockdim.x = %d, blockdim.y = %d\n", threadIdx.x, blockDim.x, blockDim.y);
	int A_index = (blockDim.y)*(blockIdx.y)*q + (threadIdx.y)*q + (blockDim.x)*(blockIdx.x) + threadIdx.x;
	int C_index = (blockDim.y)*(blockIdx.y)*q + (threadIdx.y)*q + (blockDim.x)*(blockIdx.x) + threadIdx.x;

        C[C_index] = tanhf(A[A_index]);

}

__host__ void LSTM::mat_tanh_gpu(const float *A, float *C, int p, int q){

    //max threads per block needs to taken care while using dim3. Else kernel wont launch.
    dim3 dimBlock(m_block_dim, m_block_dim, 1);
	dim3 dimGrid(((p+dimBlock.x-1)/dimBlock.x), ((q+dimBlock.y-1)/dimBlock.y), 1); // There may be some incomplete(not block_dimXblock_dim) blocks at the edges
	//kernel Launch
	mat_tanh_kernel<float><<<dimGrid, dimBlock>>>(A, C, p, q);
	     cudaDeviceSynchronize();

}
