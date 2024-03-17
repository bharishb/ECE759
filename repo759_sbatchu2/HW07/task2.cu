#include "count.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <iostream>
#include <random>
#include <cassert>
#include <thrust/copy.h>
using namespace std;

int main(int argc, char** argv){
	
    unsigned int n = atoi(argv[1]);

    //allocate host memory
    thrust::host_vector<int> h_a(n);
    thrust::host_vector<int> h_values(n);
    thrust::host_vector<int> h_counts(n);


    //allocate device memory
    thrust::device_vector<int> d_a(n);
    thrust::device_vector<int> values(n);
    thrust::device_vector<int> counts(n);

    //random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> dist(0, 500);

    for(size_t i=0; i<n; i++)
    {
        h_a[i] = dist(generator);
    }

    //copy host vector to device
    thrust::copy(h_a.begin(), h_a.end(), d_a.begin());

    //Kernel call with timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    count(d_a, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  
    //copy device vector to host
    thrust::copy(values.begin(), values.end(), h_values.begin());
    thrust::copy(counts.begin(), counts.end(), h_counts.begin());

   //printf("values size : %d\n", values.size());
   //last element of values array
   printf("%d\n", h_values[values.size()-1]);
   //printf("%d\n", values[values.size()-1]);

   //last element of counts array'
   printf("%d\n", h_counts[values.size()-1]);
   //printf("%d\n", counts[values.size()-1]);

   /*for(int i=0; i<h_a.size(); i++)
    {
        printf("h_a values %d : %d\n", i, h_a[i]);
    }
   for(int i=0; i<values.size(); i++)
    {
        printf("values %d : %d\n", i, h_values[i]);
    }

    for(int i=0; i<values.size(); i++)
    {
        printf("counts %d : %d\n", i, h_counts[i]);
    }*/
    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f\n", ms);
    //printf("Time Taken in Milliseconds : %f",ms);


    return 0;
}
