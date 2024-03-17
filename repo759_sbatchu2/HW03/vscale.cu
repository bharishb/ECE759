#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n)
{

    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    //Avoid writing into unknown locations
    if(index < n){
        b[index] = b[index]*a[index];
    }
}
