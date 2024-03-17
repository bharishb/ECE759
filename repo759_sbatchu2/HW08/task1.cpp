#include "matmul.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <ratio>
#include <vector>
using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

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


int main(int argc, char** argv){

    //read n value to create matrix nxn
    unsigned int n = atoi(argv[1]);
    //read number of threads to create parallel threads in omp
    unsigned int t = atoi(argv[2]);
    omp_set_num_threads(t); // set t number of parallel threads

    // random number generator
  std::random_device rd;       // obtain a random number
  std::mt19937 rand_gen(rd()); // seed the generator

  // Time measurement
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec; // milli is from ratio library

  float *A = new float[n * n];
  float *B = new float[n * n];
  float *C = new float[n * n];
  float *C_golden = new float[n * n];

  std::uniform_real_distribution<float> dist(-10.0, 10.0);

  for (unsigned int i = 0; i < n * n; i++) {
    A[i] = dist(rand_gen); // here dist is a  callable object
    B[i] = dist(rand_gen);
  }

  //call mmul with timing measurement
  //double start = omp_get_wtime();
  start = high_resolution_clock::now();
  mmul(A, B, C, n);
  end = high_resolution_clock::now();
  //double end = omp_get_wtime();

  // Convert the calculated duration to a double using the standard library
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);
 
  //golden reference
  matmul_golden(A,B,C_golden,n);

  //debug prints
  /*print_matrix(A,n);
  print_matrix(B,n);
  print_matrix(C,n);
  print_matrix(C_golden,n);*/
  compare_matrix(C, C_golden, n);

  //first element 
  printf("%f\n",C[0]);
  //last element
  printf("%f\n",C[n*n-1]);
  //time in milliseconds
  //printf("%.10f\n",(end-start)*1000);
  cout << duration_sec.count() << endl;
  //printf("diff  = %.16g\n", end - start);

  return 0;
}
