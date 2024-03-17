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

void print_matrix(const double *X, int n) {
  cout << "Matrix check" << endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cout << X[j + i * n] << " ";
    }
    cout << endl;
  }
  cout << endl;
}

void compare_matrix(const double *X, const double *Y, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      assert((X[i * n + j] == Y[i * n + j]));
    }
  }
}

int main() {

  int n = 1024;

  cout << n << endl;
  // Time measurement
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec; // milli is from ratio library

  // random number generator
  std::random_device rd;       // obtain a random number
  std::mt19937 rand_gen(rd()); // seed the generator

  double *A = new double[n * n];
  double *B = new double[n * n];
  double *C1 = new double[n * n];
  double *C2 = new double[n * n];
  double *C3 = new double[n * n];
  double *C4 = new double[n * n];
  vector<double> Av, Bv;

  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (int i = 0; i < n * n; i++) {
    A[i] = dist(rand_gen); // here dist is a  callable object
    B[i] = dist(rand_gen);
    Av.push_back(A[i]);
    Bv.push_back(B[i]);
  }

  // call variatians of Matrix Multiplication
  start = high_resolution_clock::now();
  mmul1(A, B, C1, n);
  end = high_resolution_clock::now();
  // Convert the calculated duration to a double using the standard library
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  // Durations are converted to milliseconds - std::chrono::duration_cast
  // cout << "Total time in convolution: " << duration_sec.count() << "ms\n";
  cout << duration_sec.count() << endl;

  start = high_resolution_clock::now();
  mmul2(A, B, C2, n);
  end = high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);
  cout << duration_sec.count() << endl;

  start = high_resolution_clock::now();
  mmul3(A, B, C3, n);
  end = high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);
  cout << duration_sec.count() << endl;

  start = high_resolution_clock::now();
  mmul4(Av, Bv, C4, n);
  end = high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);
  cout << duration_sec.count() << endl;

  // Last element prints
  cout << C1[n * n - 1] << endl;
  cout << C2[n * n - 1] << endl;
  cout << C3[n * n - 1] << endl;
  cout << C4[n * n - 1] << endl;

  compare_matrix(C1, C2, n);
  compare_matrix(C2, C3, n);
  compare_matrix(C3, C4, n);

  delete[] A;
  A = nullptr;
  delete[] B;
  B = nullptr;
  delete[] C1;
  C1 = nullptr;
  delete[] C2;
  C2 = nullptr;
  delete[] C3;
  C3 = nullptr;
  delete[] C4;
  C4 = nullptr;

  return 0;
}
