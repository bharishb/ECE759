#include <cuda.h>
#include "matmul.cuh"
#include <stdio.h>
#include <iostream>
#include <random>
#include <cassert>
using namespace std;

template<typename T>
void print_matrix(const T *X, size_t n) {
  cout << "Matrix check" << endl;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      cout << X[j + i * n] << " ";
    }
    cout << endl;
  } 
  cout << endl;
}   

template<typename T>
void matmul_golden(const T*A, const T* B, T*C, size_t n){
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

template<typename T>
void compare_matrix(const T* A, const T* B, size_t n)
{
    for(size_t i=0; i<n; i++) {
	    for(size_t j=0; j<n; j++) {
		    assert(abs(A[i+j*n]-B[i+j*n])<5);
		    /*if(abs(A[i+j*n]-B[i+j*n])>5) {
		           printf("MISMATCH golden value = %f, observed value = %f\n",A[i+j*n], B[i+j*n]);
		     return;}*/
	    }
    }
}


template<typename T>
void launch_matmul_kernel(size_t n, unsigned int block_dim, double* hA_double, double* hB_double){

    T* hA;
    T* hB;
    T* hC;
    T* hC_golden;

    hA = new T[n*n]; 
    hB = new T[n*n]; 
    hC = new T[n*n]; 
    hC_golden = new T[n*n];


    for(unsigned long long int i=0; i<n*n ; i++){
        hA[i] = (T) hA_double[i];
        hB[i] = (T) hB_double[i];
    }

    //device memory
    //matmul_1
    T* dA;
    T* dB;
    T* dC;   //device memory

    //Allocate device memory
    cudaMalloc((void**)&dA, sizeof(T)*n*n);
    cudaMalloc((void**)&dB, sizeof(T)*n*n);
    cudaMalloc((void**)&dC, sizeof(T)*n*n);

    //Move to device memory
    cudaMemcpy(dA, hA, sizeof(T)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(T)*n*n, cudaMemcpyHostToDevice);

    //Kernel call with timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (std::is_same<T, int>::value){
        matmul_1((int*)dA, (int*)dB, (int*)dC, n, block_dim);
    } else if (std::is_same<T, float>::value){
        matmul_2((float*)dA, (float*)dB, (float*)dC, n, block_dim);
    } else {
        matmul_3((double*)dA, (double*)dB, (double*)dC, n, block_dim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Move from device memory to host memory
    cudaMemcpy(hC, dC, sizeof(T)*n*n, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    //printf("Time Taken in Milliseconds : %f",ms);

    if (std::is_same<T, int>::value){
      //first element
      printf("%d\n",(int)hC[0]);
      //last element print
      printf("%d\n",(int)hC[n*n-1]);
    } else if (std::is_same<T, float>::value){
      //first element
      printf("%f\n",(float)hC[0]);
      //last element print
      printf("%f\n",(float)hC[n*n-1]);
    } else {
      //first element
      printf("%f\n",(double)hC[0]);
      //last element print
      printf("%f\n",(double)hC[n*n-1]);
    }
    printf("%f\n", ms);
    //matmul_golden<T>(hA, hB, hC_golden, n);
    //compare_matrix<T>(hC_golden, hC, n);
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
    delete [] hC_golden;
    hC_golden = nullptr;
}

int main(int argc, char** argv)
{
    size_t n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]);
   
    //trying to give same matrix data with various datatypes
    double* hA_double;
    double* hB_double;

    hA_double = new double[n*n];
    hB_double = new double[n*n];

    //random number generator
    std::random_device rd;
    std::mt19937 generator(rd());

    std::uniform_real_distribution<double> dista(-1000,1000);
    std::uniform_real_distribution<double> distb(-1000,1000);

    for(size_t i=0; i<n*n; i++)
    {
      hA_double[i] = dista(generator);
      hB_double[i] = distb(generator);
    }

    //matmul_1
    launch_matmul_kernel<int>(n, block_dim, hA_double, hB_double);

    //matmul_2
    launch_matmul_kernel<float>(n, block_dim, hA_double, hB_double);

    //matmul_3
    launch_matmul_kernel<double>(n, block_dim, hA_double, hB_double);
}
