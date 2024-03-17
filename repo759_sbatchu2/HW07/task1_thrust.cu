#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <iostream>
#include <random>
#include <cassert>
using namespace std;

void reduce_golden(const float* X, float* sum, unsigned int n){
    *sum = 0.0;
    for(unsigned int i=0; i<n; i++){
        *sum+=X[i];
    }
}

int main(int argc, char** argv){
	
    unsigned int n = atoi(argv[1]);
    float sum;
    float sum_golden;
    //allocate host memory
    thrust::host_vector<float> h_a(n);

    //random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dist(-1,1);

    for(size_t i=0; i<n; i++)
    {
        h_a[i] = dist(generator);
    }

    //allocate device memory
    thrust::device_vector<float> d_a(n);

    //copy host vector to device
    d_a = h_a;

    //Kernel call with timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //reduce kernel
    sum = thrust::reduce(d_a.begin(), d_a.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //reduce_golden
    reduce_golden(&h_a[0], &sum_golden, n);

    //assert
    assert(abs(sum_golden - sum)<0.05);

    //prints
    //printf("sum_golden : %f\n", sum_golden);
    //printf("sum_gpu : %f\n", sum);
    printf("%f\n", sum);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f\n", ms);
    //printf("Time Taken in Milliseconds : %f",ms);


    return 0;
}
