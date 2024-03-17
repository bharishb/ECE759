
#include <cuda.h>
#include "LSTM.cuh"
__global__ void mat_add_kernel(const float* A, const float* B, float* C, int p, int q) {

        //Addition by reading from Global Memory
printf("kernel call mat_add_kernel : %d, blockdim.x = %d, blockdim.y = %d\n", threadIdx.x, blockDim.x, blockDim.y);

	int A_index = (blockDim.y)*(blockIdx.y)*q + (threadIdx.y)*q + (blockDim.x)*(blockIdx.x) + threadIdx.x;
	int B_index = (blockDim.y)*(blockIdx.y)*q + (threadIdx.y)*q + (blockDim.x)*(blockIdx.x) + threadIdx.x;
	int C_index = (blockDim.y)*(blockIdx.y)*q + (threadIdx.y)*q + (blockDim.x)*(blockIdx.x) + threadIdx.x;

        C[C_index] = A[A_index] + B[B_index];

}

__host__ void LSTM::mat_add_gpu(const float *A, const float *B, float *C, int p, int q){

    //max threads per block needs to taken care while using dim3. Else kernel wont launch.
    dim3 dimBlock(m_block_dim, m_block_dim, 1);
	dim3 dimGrid(((p+m_block_dim-1)/m_block_dim), ((q+m_block_dim-1)/m_block_dim), 1); // There may be some incomplete(not block_dimXblock_dim) blocks at the edges
	//kernel Launch
	mat_add_kernel<<<dimGrid, dimBlock>>>(A, B, C, p, q);
	     cudaDeviceSynchronize();

}
