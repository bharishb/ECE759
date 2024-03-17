#include <cuda.h>
#include "matmul.cuh"
#include <stdio.h>
#include <iostream>
using namespace std;

//Tiling is done across shared memories present in all SMs. 
//Each shared memory will get block_dim X block_dim amount of data each of A, B and compute C tile corresponding to them.
template<typename T>
__global__ void matmul_kernel(const T* A, const T* B, T*C, unsigned int n){

//Dynamic shared memory declared
extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
T *shared_mem = reinterpret_cast<T *>(my_smem);

//assigning shared memory to tiled A,B,C pointers
T* tileA = (T*)shared_mem;
T* tileB = (T*)&shared_mem[blockDim.x*blockDim.y];
//T* tileC = (T*)&s[2*blockDim.x*blockDim.y];

T dot_sum = 0;
for(unsigned int i=0; i<((n+blockDim.x-1)/blockDim.x); i++)
{
	unsigned long int A_index = blockIdx.y*n*blockDim.y+i*blockDim.x+threadIdx.x+threadIdx.y*n;
	unsigned int A_index_x = i*blockDim.x+threadIdx.x;
	unsigned int A_index_y = blockIdx.y*blockDim.y + threadIdx.y;
	if((A_index < n*n) && (A_index_x < n) && (A_index_y < n))
	    tileA[threadIdx.x+blockDim.x*threadIdx.y] = A[A_index];
    else 
	    tileA[threadIdx.x+blockDim.x*threadIdx.y] = 0;

	//printf("tile A thread x : %d, thread y : %d, value : %f\n", threadIdx.x, threadIdx.y,  tileA[threadIdx.x+blockDim.x*threadIdx.y]);
	
	unsigned long int B_index = blockIdx.x*blockDim.x+i*n*blockDim.y+threadIdx.x+threadIdx.y*n;
    unsigned int B_index_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int B_index_y = i*blockDim.y + threadIdx.y;
	if((B_index < n*n) && (B_index_x < n) && (B_index_y < n))
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
    
	unsigned long int C_index = blockIdx.y*blockDim.y*n+threadIdx.y*n+blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int C_index_x = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int C_index_y = blockIdx.y*blockDim.y+threadIdx.y;
    if((C_index < n*n) && (C_index_x < n) && (C_index_y < n)) {
		C[C_index] = dot_sum;
	} 

}





__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){

    dim3 dimBlock(block_dim, block_dim);
	dim3 dimGrid((n+block_dim-1)/block_dim, (n+block_dim-1)/block_dim); // There may be some incomplete(not block_dimXblock_dim) blocks at the edges

	//kernel Launch
	matmul_kernel<int><<<dimGrid, dimBlock, 2*(block_dim*block_dim)*sizeof(int)>>>(A, B, C, n);

}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){

    //max threads per block needs to taken care while using dim3. Else kernel wont launch.
    dim3 dimBlock(block_dim, block_dim, 1);
	dim3 dimGrid(((n+dimBlock.x-1)/dimBlock.x), ((n+dimBlock.y-1)/dimBlock.y), 1); // There may be some incomplete(not block_dimXblock_dim) blocks at the edges
	//kernel Launch
	matmul_kernel<<<dimGrid, dimBlock, 2*(block_dim*block_dim)*sizeof(float)>>>(A, B, C, n);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){

    dim3 dimBlock(block_dim, block_dim);
	dim3 dimGrid((n+block_dim-1)/block_dim, (n+block_dim-1)/block_dim); // There may be some incomplete(not block_dimXblock_dim) blocks at the edges

	//kernel Launch
	matmul_kernel<double><<<dimGrid, dimBlock, 2*(block_dim*block_dim)*sizeof(double)>>>(A, B, C, n);

}
