
#include <cuda.h>
#include "LSTM.cuh"

//This Kernel resets all network states
template<typename T>
__global__ void reset_all_network_states_kernel(T* A, T* B, T* C, T*D, int n) {  // n is size

        //Addition by reading from Global Memory

	
	int thread_index = (blockIdx.x)*(blockDim.x) + threadIdx.x;

	if(thread_index < n) {
           A[thread_index] = 0;;
           B[thread_index] = 0;;
           C[thread_index] = 0;;
           D[thread_index] = 0;;
	}
}
 void LSTM::reset_all_network_states_gpu()
{
    //max threads per block needs to taken care while using dim3. Else kernel wont launch.
    dim3 dimBlock(m_block_dim*m_block_dim, 1, 1);
	dim3 dimGrid(((m_batch_size*m_hidden_size + m_block_dim*m_block_dim -1)/m_block_dim*m_block_dim), 1, 1); // There may be some incomplete(not block_dimXblock_dim) blocks at the edges
	//kernel Launch
	reset_all_network_states_kernel<float><<<dimGrid, dimBlock>>>(c_t_gpu, h_t_gpu, c_t_minus_1_gpu, h_t_minus_1_gpu, m_batch_size*m_hidden_size);
	     cudaDeviceSynchronize();
}
