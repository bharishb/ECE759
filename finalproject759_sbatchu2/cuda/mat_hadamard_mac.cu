
#include <cuda.h>
#include "LSTM.cuh"
template<typename T>
__global__ void mat_hadamard_mac_kernel(const T* A, const T* B, T* C, int p, int q, int accum_mode) {

        //Addition by reading from Global Memory

printf("kernel call hadamard mac: %d, blockdim.x = %d, blockdim.y = %d\n", threadIdx.x, blockDim.x, blockDim.y);
	int A_index = (blockDim.y)*(blockIdx.y)*q + (threadIdx.y)*q + (blockDim.x)*(blockIdx.x) + threadIdx.x;
	int B_index = (blockDim.y)*(blockIdx.y)*q + (threadIdx.y)*q + (blockDim.x)*(blockIdx.x) + threadIdx.x;
	int C_index = (blockDim.y)*(blockIdx.y)*q + (threadIdx.y)*q + (blockDim.x)*(blockIdx.x) + threadIdx.x;

	if(accum_mode == 1)
            C[C_index] += A[A_index] * B[B_index];
	else
            C[C_index] = A[A_index] * B[B_index];

}

__host__ void LSTM::mat_hadamard_mac_gpu(const float *A, const float *B, float *C, int p, int q){

    //max threads per block needs to taken care while using dim3. Else kernel wont launch.
    dim3 dimBlock(m_block_dim, m_block_dim, 1);
	dim3 dimGrid(((p+dimBlock.x-1)/dimBlock.x), ((q+dimBlock.y-1)/dimBlock.y), 1); // There may be some incomplete(not block_dimXblock_dim) blocks at the edges
	//kernel Launch
	mat_hadamard_mac_kernel<float><<<dimGrid, dimBlock>>>(A, B, C, p, q, 1);
	     cudaDeviceSynchronize();

}
