
#include <cuda.h>
#include "LSTM.cuh"

//This Kernel removes stride accesses in input by moving data to a buffer.
__global__ void buffer_batch_input_kernel(const float* A, float* C, int offset, int n1, int n2, int stride) {  // n1 is Batch size, n2 is input size : extracting batch size inputs of each input_size amount at seq_length strides

        //Addition by reading from Global Memory

	
	int thread_index = (blockIdx.x)*(blockDim.x) + threadIdx.x;
	int A_index = (thread_index/n2)*stride + thread_index % n2 + offset;
	int C_index = thread_index;

	if((A_index < (n1*stride + n2)) && (C_index <n1*n2))  // need to handle input size
           C[C_index] = A[A_index];
    printf("Buffer kernel code threadIdx : %d, A[%d] = %f , C[%d] = %f\n", threadIdx.x, A_index, A[A_index], C_index, C[C_index]);
}

void LSTM::move_gpu_input_data_to_buffer(const float *A, int offset, float *C)
{
    printf("Buffer kernel before start\n");
    //max threads per block needs to taken care while using dim3. Else kernel wont launch.
    dim3 dimBlock(m_block_dim*m_block_dim, 1, 1);
	dim3 dimGrid(((m_batch_size*m_input_size + m_block_dim*m_block_dim -1)/(m_block_dim*m_block_dim)), 1, 1); // There may be some incomplete(not block_dimXblock_dim) blocks at the edges
	//kernel Launch
	buffer_batch_input_kernel<<<dimGrid, dimBlock>>>(A, C, offset, m_batch_size, m_input_size, m_seq_length*m_input_size);
    cudaDeviceSynchronize();
    printf("Buffer kernel after end\n");
}
