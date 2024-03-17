#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include "vscale.cuh"
#define NUM_THREADS_PER_BLOCK 16
using namespace std;

int main(int argc, char**argv){
    char* n_string = argv[1]; // First argument, 0 is executable
    unsigned int  n = atoi(n_string);

    //random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dista(-10.0,10.0);
    std::uniform_real_distribution<float> distb(0,1.0);

    float* host_a;
    float* host_b;
    float* device_a;
    float* device_b;
    host_a = new float[n];
    host_b = new float[n];

    for(unsigned int i=0; i<n; i++)
    {
        host_a[i] = dista(generator);
        host_b[i] = distb(generator);
    }

    /*//print input arrays
    printf("Array a\n");
    for(int i=0; i<n; i++) printf("%f ",host_a[i]);
    printf("\nArray b\n");
    for(int i=0; i<n; i++) printf("%f ",host_b[i]);;
    printf("\nArray Expected output\n");
    for(int i=0; i<n; i++) printf("%f ",host_b[i]*host_a[i]);
    printf("\n");*/

    //Allocate device Memory
    cudaMalloc((void**)&device_a, sizeof(float)*n);
    cudaMalloc((void**)&device_b, sizeof(float)*n);
    cudaMemcpy(device_a, host_a, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, sizeof(float)*n, cudaMemcpyHostToDevice);

    //Kernel call with timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vscale<<<(n+NUM_THREADS_PER_BLOCK-1)/NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>(device_a, device_b, n); // kernel called
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(host_b, device_b, sizeof(float)*n, cudaMemcpyDeviceToHost);
    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    //printf("Time Taken in Milliseconds : %f",ms);
    printf("%f\n", ms);
    printf("%f\n",host_b[0]);
    printf("%f\n",host_b[n-1]);

    /*printf("Array b output \n");
    for(int i=0; i<n; i++) printf("%f ",host_b[i]);*/
    
    return 0;
}
