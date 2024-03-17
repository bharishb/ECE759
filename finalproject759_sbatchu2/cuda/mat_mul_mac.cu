#include <cuda.h>
#include "LSTM.cuh"
using namespace std;

//Tiling is done across shared memories present in all SMs. 
//Each shared memory will get m_block_dim X m_block_dim amount of data each of A, B and compute C tile corresponding to them.
//standard (m,n,k) = (p,r,q) : k/q dimension is accumulating dimension for each output tile.
template<typename T>
__global__ void mat_mul_mac_kernel(const T* A, const T* B, T*C, unsigned int p, unsigned int q, unsigned int r, int accum_mode){

//Dynamic shared memory declared
extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
T *shared_mem = reinterpret_cast<T *>(my_smem);

//assigning shared memory to tiled A,B,C pointers
T* tileA = (T*)shared_mem;
T* tileB = (T*)&shared_mem[blockDim.x*blockDim.y];
//T* tileC = (T*)&s[2*blockDim.x*blockDim.y];

printf("kernel call mat_mul_mac : %d, blockdim.x = %d, blockdim.y = %d\n", threadIdx.x, blockDim.x, blockDim.y);
T dot_sum = 0;
for(unsigned int i=0; i<((q+blockDim.x-1)/blockDim.x); i++)
{
	unsigned long int A_index = blockIdx.y*q*blockDim.y+i*blockDim.x+threadIdx.x+threadIdx.y*q;
	unsigned int A_index_x = i*blockDim.x+threadIdx.x;
	unsigned int A_index_y = blockIdx.y*blockDim.y + threadIdx.y;
	if((A_index < p*q) && (A_index_x < q) && (A_index_y < p))
	    tileA[threadIdx.x+blockDim.x*threadIdx.y] = A[A_index];
    else 
	    tileA[threadIdx.x+blockDim.x*threadIdx.y] = 0;

	//printf("tile A thread x : %d, thread y : %d, value : %f\n", threadIdx.x, threadIdx.y,  tileA[threadIdx.x+blockDim.x*threadIdx.y]);
	
	unsigned long int B_index = blockIdx.x*blockDim.x+i*r*blockDim.y+threadIdx.x+threadIdx.y*r;
    unsigned int B_index_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int B_index_y = i*blockDim.y + threadIdx.y;
	if((B_index < q*r) && (B_index_x < r) && (B_index_y < q))
	    tileB[threadIdx.x+blockDim.x*threadIdx.y] = B[B_index];
    else
	    tileB[threadIdx.x+blockDim.x*threadIdx.y] = 0;
    //printf("tile B thread x : %d, thread y : %d, value : %f\n", threadIdx.x, threadIdx.y,  tileB[threadIdx.x+blockDim.x*threadIdx.y]);

	//sync to make sure data is reached to shared memory
	__syncthreads();

   

	for(int k =0; k<blockDim.x; k++)
	{
         dot_sum += tileA[k+threadIdx.y*blockDim.x]*tileB[threadIdx.x + k*blockDim.x];
	}
	//printf("tile C thread x : %d, thread y : %d, value : %f\n", threadIdx.x, threadIdx.y, dot_sum);
	__syncthreads();
}
    
	unsigned long int C_index = blockIdx.y*blockDim.y*r+threadIdx.y*r+blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int C_index_x = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int C_index_y = blockIdx.y*blockDim.y+threadIdx.y;
    if((C_index < p*r) && (C_index_x < r) && (C_index_y < p)) {
	        if(accum_mode == 1) {
		   C[C_index] += dot_sum;
		} else {
		   C[C_index] = dot_sum;
		}
	} 

}


void LSTM::mat_mac_gpu(const float *A, const float *B, float *C, int p, int q, int r){

    printf("gpu about to call kernel1 mat_mul_mac , block_dim : %d, p : %d, q : %d, r : %d\n", m_block_dim, p, q, r);
    //max threads per block needs to taken care while using dim3. Else kernel wont launch.
    dim3 dimBlock(m_block_dim, m_block_dim, 1);
	dim3 dimGrid(((p+dimBlock.x-1)/dimBlock.x), ((r+dimBlock.y-1)/dimBlock.y), 1); // There may be some incomplete(not block_dimXblock_dim) blocks at the edges
	//kernel Launch
	mat_mul_mac_kernel<float><<<dimGrid, dimBlock, 2*(m_block_dim*m_block_dim)*sizeof(float)>>>(A, B, C, p, q, r, 1);
    cudaDeviceSynchronize();
    printf("gpu about to call kernel2 mat_mul_mac , block_dim : %d, p : %d, q : %d, r : %d\n", m_block_dim, p, q, r);
}

