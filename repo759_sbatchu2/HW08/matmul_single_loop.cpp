#include<iostream>
#include "matmul.h"

using namespace std;

void mmul(const float* A, const float* B, float* C, const std::size_t n){
  
  #pragma omp parallel for
  for (unsigned int index = 0; index < n*n; index++) {
    for (unsigned int k = 0; k < n; k++) {

        unsigned int i = index/n;
        unsigned int j = index%n;

        //print thread id
        //int myId = omp_get_thread_num();
        //printf("I am thread No.  %d (i, k, j) = (%d, %d, %d)\n",myId, i, k, j);

        if (k == 0) {
          C[index] = 0; // initialization
        }
        C[index] = C[index] + A[i * n + k] * B[k * n + j]; // dot
                                                                   // product
    }
  }

}


