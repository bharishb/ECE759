#include <cuda.h>
#include "reduce.cuh"
#include <stdio.h>
#include <iostream>
#include <random>
#include <cassert>
using namespace std;

void print_vector(const float *X, size_t n) {
  cout << "Vector check" << endl;
    for (size_t j = 0; j < n; j++) {
      cout << X[j] << " ";
    }
    cout << endl;
}   

void reduce_golden(const float* X, float* sum, unsigned int n){
    *sum =0.0;
    for(unsigned int i=0; i<n; i++){
        *sum+=X[i];
    }
}

int main(int argc, char** argv){

unsigned int n = atoi(argv[1]);
unsigned int threads_per_block = atoi(argv[2]);
    
float* h_input;
float* d_input;
float sum_golden;

h_input = new float[n];

//random number generator
std::random_device rd;
std::mt19937 generator(rd());

std::uniform_real_distribution<float> dist(-1,1);

for(size_t i=0; i<n; i++)
{
    h_input[i] = dist(generator);
}

//golden
//print_vector(h_input, n);
reduce_golden(h_input,&sum_golden, n);
//cout<<"sum :"<<sum_golden<<endl;

//Allocate device memory
cudaMalloc((void**)&d_input, sizeof(float)*n);

//Move to device memory
cudaMemcpy(d_input, h_input, sizeof(float)*n, cudaMemcpyHostToDevice);

//Kernel call with timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce(&d_input, &d_input, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Move from device memory to host memory
    cudaMemcpy(h_input, d_input, sizeof(float)*n, cudaMemcpyDeviceToHost);

    cudaFree(d_input);

    //GPU sum
    //cout <<"GPU Sum : "<<h_input[0]<<endl;
    cout <<h_input[0]<<endl;
    assert(abs(sum_golden-h_input[0])<0.5);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f\n", ms);  
    //printf("Time Taken in Milliseconds : %f",ms);

    delete [] h_input;
    h_input = nullptr;

return 0;

}

