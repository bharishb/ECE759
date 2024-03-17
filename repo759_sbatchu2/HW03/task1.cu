#include <iostream>
#include<stdio.h>
#include <cuda.h>
using namespace std;

__global__ void factorial()
{
    int fac;
    for(int i=1; i<=(threadIdx.x+1); i++)
    {
        fac = (i==1) ? 1 : fac*i ;
    }
    printf("%d!=%d\n",(threadIdx.x+1),fac);
}
int main()
{
   
    factorial<<<1,8>>>(); // 1 block, 8 threads per block
    cudaDeviceSynchronize();
    return 0;
}
