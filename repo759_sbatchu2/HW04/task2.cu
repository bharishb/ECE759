#include <cuda.h>
#include "stencil.cuh"
#include <stdio.h>
#include <iostream>
#include <random>
#include <cassert>
using namespace std;

void print_matrix(const float *X, int n) {
  cout << "Matrix check" << endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cout << X[j + i * n] << " ";
    }
    cout << endl;
  }
  cout << endl;
}

void print_vector(const float *X, int n){
   cout << "Vector check" << endl;
   for(int i=0; i<n; i++)
   {
       cout << X[i] << " ";
   } 
   cout << endl;
}

void stencil_golden(const float* a, const float*b, float*c, unsigned int n, unsigned int R)
{
    for(unsigned int i=0; i<n; i++)
    { 
	    c[i] = 0.0;
	    for(unsigned int j=0; j<(2*R+1); j++) {
                int index = i + j - R;
	        c[i] += (((index<0) || (index>(long int)(n-1))) ? 1 : a[index]) *b[j];
	    }
    }

}	

void compare_vector(const float* X, const float* Y, unsigned int n)
{ 
	for(unsigned int i=0; i<n; i++)
	{
            assert(abs(X[i]-Y[i])<0.002);
	    /*if(abs(X[i]-Y[i])>0.01)
	    {
	        printf("Mismatch seen. X[%d]=%f, Y[%d]=%f\n",i, X[i], i, Y[i]);
	    }*/
	}
}

int main(int argc, char** argv)
{
    size_t n = atoi(argv[1]);
    size_t R = atoi(argv[2]);
    unsigned int threads_per_block = atoi(argv[3]);

    float* h_image;
    float* h_mask;
    float* h_output;
    float* h_output_golden;

    h_image = new float[n]; 
    h_mask = new float[2*R+1]; 
    h_output = new float[n]; 
    h_output_golden = new float[n]; 

    //random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dist_image(-1,1);
    std::uniform_real_distribution<float> dist_mask(-1,1);

    for(size_t i=0; i<n; i++)
    {
        h_image[i] = dist_image(generator);
    }

    for(size_t i=0; i<(2*R+1); i++)
    {
        h_mask[i] = dist_mask(generator);
    }


    float* d_image;
    float* d_mask;
    float* d_output;

    //global memory allocation
    cudaMalloc((void**)&d_image, sizeof(float)*n);
    cudaMalloc((void**)&d_mask, sizeof(float)*(2*R+1));
    cudaMalloc((void**)&d_output, sizeof(float)*n);
    
    //moving data to device
    cudaMemcpy(d_image, h_image, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, sizeof(float)*(2*R+1), cudaMemcpyHostToDevice);

    //Kernel call with timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_output, d_output, sizeof(float)*n, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    //printf("Time Taken in Milliseconds : %f",ms);
    //last element print
    printf("%f\n",h_output[n-1]);
    printf("%f\n", ms);

    //check 
    stencil_golden(h_image, h_mask, h_output_golden, n, R);
    compare_vector(h_output_golden, h_output, n);

    delete [] h_image;
    h_image = nullptr;
    delete [] h_mask;
    h_mask = nullptr;
    delete [] h_output;
    h_output = nullptr;

    return 0;

}
