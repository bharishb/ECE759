#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <random>
using namespace std;
#define NUM_THREADS_PER_BLOCK 8
#define NUM_ELEMS 16

__global__ void saxpy_int_example(int* dA, int a)
{
    int index = blockIdx.x*NUM_THREADS_PER_BLOCK + threadIdx.x;
    //Avoid writing into unknown locations
    if(index < NUM_ELEMS){
        dA[index] = a*threadIdx.x + blockIdx.x;
    }
}

int main()
{ 
    int hA[NUM_ELEMS];  //host array
    int* dA;            //Device array

    //random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> dist(-1000,1000);
    int a = dist(generator);

    cudaMalloc((void**)&dA, sizeof(int)*NUM_ELEMS);
    saxpy_int_example<<<NUM_ELEMS/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(dA,a); // calling the kernel with 2 thread blocks with 8 elements each
    cudaMemcpy(hA, dA, NUM_ELEMS*(sizeof(int)), cudaMemcpyDeviceToHost); // dont need explicit cudaDeviceSynchronize
    for(int i=0; i<NUM_ELEMS; i++){
        printf("%d ",hA[i]);
    }

}
