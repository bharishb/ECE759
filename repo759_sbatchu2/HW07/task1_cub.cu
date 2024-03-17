#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
#include <iostream>
#include <random>
#include <cassert>
using namespace std;
using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

void reduce_golden(const float* X, float* sum, unsigned int n){
    *sum = 0.0;
    for(unsigned int i=0; i<n; i++){
        *sum+=X[i];
    }
}

int main(int argc, char** argv){
	
    unsigned int n = atoi(argv[1]);
    float gpu_sum;
    float sum_golden;

    //allocate host memory
    float* h_in;
    h_in = new float[n];

    //device memory pointer
    float* d_in;

    //random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dist(-1,1);

    for(size_t i=0; i<n; i++)
    {
        h_in[i] = dist(generator);
    }

    //allocate device memory
    g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * n);

    //copy host vector to device
    cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);

    float* d_sum = NULL;
    g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1);

    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    //Kernel call with timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //reduce kernel
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost);

    //reduce_golden
    reduce_golden(&h_in[0], &sum_golden, n);

    //assert
    assert(abs(sum_golden - gpu_sum)<0.05);

    //prints
    //printf("sum_golden : %f\n", sum_golden);
    //printf("sum_gpu : %f\n", gpu_sum);
    printf("%f\n", gpu_sum);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f\n", ms);
    //printf("Time Taken in Milliseconds : %f",ms);

    //free up memory
    g_allocator.DeviceFree(d_in);
    g_allocator.DeviceFree(d_sum);
    g_allocator.DeviceFree(d_temp_storage);

    return 0;
}
