#include "matmul.h"

// i, j, k -> rows of C, columns of C, dot product dimension respectively
// inner loop k -> so A is accessed cache friendly
void mmul1(const double *A, const double *B, double *C, const unsigned int n) {
  for (unsigned int i = 0; i < n; i++) // rows
  {
    for (unsigned int j = 0; j < n; j++) // columns
    {
      for (unsigned int k = 0; k < n; k++) {
        if (k == 0) {
          C[j + i * n] = 0.0; // initialization
        }
        C[j + i * n] = C[j + i * n] + A[i * n + k] * B[k * n + j]; // dot
                                                                   // product
      }
    }
  }
}

// i, k, j -> rows of C, dot product dimension, columns of C
// inner loop j -> B, C are accessed cache friendly
void mmul2(const double *A, const double *B, double *C, const unsigned int n) {
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int k = 0; k < n; k++) {
      for (unsigned int j = 0; j < n; j++) {
        if (k == 0) {
          C[j + i * n] = 0; // initialization
        }
        C[j + i * n] = C[j + i * n] + A[i * n + k] * B[k * n + j]; // dot
                                                                   // product
      }
    }
  }
}

// j, k, i
// inner loop i -> None of the accesses are cache friendly
void mmul3(const double *A, const double *B, double *C, const unsigned int n) {
  for (unsigned int j = 0; j < n; j++) {
    for (unsigned int k = 0; k < n; k++) {
      for (unsigned int i = 0; i < n; i++) {
        if (k == 0) {
          C[j + i * n] = 0; // initialization
        }
        C[j + i * n] = C[j + i * n] + A[i * n + k] * B[k * n + j]; // dot
                                                                   // product
      }
    }
  }
}

// i, j, k but with vectors
// inner loop k -> A is accessed cache friendly
void mmul4(const std::vector<double> &A, const std::vector<double> &B,
           double *C, const unsigned int n) {
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      for (unsigned int k = 0; k < n; k++) {
        if (k == 0) {
          C[j + i * n] = 0; // initialization
        }
        C[j + i * n] = C[j + i * n] + A[i * n + k] * B[k * n + j]; // dot
                                                                   // product
      }
    }
  }
}
