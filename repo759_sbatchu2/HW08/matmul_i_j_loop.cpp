#include<iostream>
#include "matmul.h"

using namespace std;

void mmul(const float* A, const float* B, float* C, const std::size_t n){
  
  #pragma omp parallel for
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int k = 0; k < n; k++) {
        for (unsigned int j = 0; j < n; j++) {

        //print thread id
        //int myId = omp_get_thread_num();
        //printf("I am thread No.  %d (i, k, j) = (%d, %d, %d)\n",myId, i, k, j);

        if (k == 0) {
          C[i*n + j] = 0; // initialization
        }
        C[i*n + j] = C[i*n + j] + A[i * n + k] * B[k * n + j]; // dot product
        
        }
    }
  }

}


