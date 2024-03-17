#include <math.h>
#include <cuda.h>
#include "LSTM.cuh"

template<typename T>
__global__ void mat_sgm_kernel(const T* A, T* C, int p, int q) {

        //Addition by reading from Global Memory
printf("kernel call sgm : %d, blockdim.x = %d, blockdim.y = %d\n", threadIdx.x, blockDim.x, blockDim.y);

	int A_index = (blockIdx.x)*(blockDim.x)+ threadIdx.x;
	int C_index = (blockIdx.x)*(blockDim.x)+ threadIdx.x;

        C[C_index] = 1/(1 + expf(-A[A_index]));

}

__host__ void LSTM::mat_sgm_gpu(const float *A, float *C, int p, int q){

    printf("Sgm kernel before start\n");
    //max threads per block needs to taken care while using dim3. Else kernel wont launch.
    dim3 dimBlock(m_block_dim*m_block_dim, 1);
	dim3 dimGrid(((p*q+m_block_dim*m_block_dim-1)/(m_block_dim*m_block_dim)), 1, 1); // There may be some incomplete(not block_dimXblock_dim) blocks at the edges
	//kernel Launch
	mat_sgm_kernel<float><<<dimGrid, dimBlock>>>(A, C, p, q);
	     cudaDeviceSynchronize();
    printf("Sgm kernel after end \n");

}
