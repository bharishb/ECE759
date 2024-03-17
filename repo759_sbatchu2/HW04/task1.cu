#include <cuda.h>
#include "matmul.cuh"
#include <stdio.h>
#include <iostream>
#include <random>
#include <cassert>
using namespace std;

void print_matrix(const float *X, int n) {
  cout << "Matrix check" << endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cout << (int)(X[j + i * n]*1000) << " ";
    }
    cout << endl;
  } 
  cout << endl;
}   

void matmul_golden(const float*A, const float* B, float*C, size_t n){
    for(size_t i=0; i<n; i++){
	    for(size_t k=0; k<n; k++){
		    for(size_t j=0; j<n; j++){
                        if (k == 0) {
                          C[j + i * n] = 0; // initialization
                        }
                        C[j + i * n] = C[j + i * n] + A[i * n + k] * B[k * n + j]; // dot
		    
		    }
	    }
    }

}

void compare_matrix(const float* A, const float* B, size_t n)
{
    for(size_t i=0; i<n; i++) {
	    for(size_t j=0; j<n; j++) {
		    assert(abs(A[i+j*n]-B[i+j*n])<0.002);
	            /*if(((int)(A[i+j*n]*100)!=(int)(B[i+j*n]*100))) {
		           printf("MISMATCH golden value = %f, observed value = %f\n",A[i+j*n], B[i+j*n]);
		    }*/
	    }
    }
}

int main(int argc, char** argv)
{
    size_t n = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    float* hA;
    float* hB;
    float* hC;
    //float* hC_golden;

    hA = new float[n*n]; 
    hB = new float[n*n]; 
    hC = new float[n*n]; 
    //hC_golden = new float[n*n]; 

    //random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dista(-1,1);
    std::uniform_real_distribution<float> distb(-1,1);

    for(size_t i=0; i<n*n; i++)
    {
        hA[i] = dista(generator);
        hB[i] = distb(generator);
    }

    //device memory
    float* dA;
    float* dB;
    float* dC;   //device memory

    //Allocate device memory
    cudaMalloc((void**)&dA, sizeof(float)*n*n);
    cudaMalloc((void**)&dB, sizeof(float)*n*n);
    cudaMalloc((void**)&dC, sizeof(float)*n*n);

    //Move to device memory
    cudaMemcpy(dA, hA, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    //Kernel call with timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul(dA, dB, dC, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Move from device memory to host memory
    cudaMemcpy(hC, dC, sizeof(float)*n*n, cudaMemcpyDeviceToHost);


    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    //printf("Time Taken in Milliseconds : %f",ms);
    //last element print
    printf("%f\n",hC[n*n-1]);
    printf("%f\n", ms);
    //matmul_golden(hA, hB, hC_golden, n);
    //compare_matrix(hC, hC_golden, n);
    /*print_matrix(hA,n);
    print_matrix(hB,n);
    print_matrix(hC,n);
    print_matrix(hC_golden,n);*/
    delete [] hA;
    hA = nullptr;
    delete [] hB;
    hB = nullptr;
    delete [] hC;
    hC = nullptr;

    return 0;
}
