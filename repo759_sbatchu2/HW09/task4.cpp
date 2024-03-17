#include "convolve.h"
#include <cassert>
#include <iostream>
#include <random>
#include <chrono>
#include <ratio>

using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

void convolution_golden(const float* image, float* output, int n, const float* mask, int m)
{
   int i_index; 
   int j_index;
   for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
        output[j + i*n] = 0.0;
         for(int im=0; im<m ; im++){
            for(int jm=0; jm<m; jm++){
               i_index = i + im - (m-1)/2;
               j_index = j + jm -(m-1)/2;
               output[j + i*n] += ((((i_index >= 0) && (i_index<n)) && ((j_index >= 0) && (j_index<n))) ? image[(i_index)*n + (j_index)] : ((((i_index >= 0) && (i_index<n)) || ((j_index >= 0) && (j_index<n))) ? 1 : 0))*mask[im*m + jm];
            }
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

int main(int argc, char** argv){

    //read n value to create matrix nxn
    unsigned int n = atoi(argv[1]);

    //mask shape
    unsigned int m = 3;

    // random number generator
    std::random_device rd;       // obtain a random number
    std::mt19937 rand_gen(rd()); // seed the generator

    // Time measurement
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec; // milli is from ratio library

  // float array memory : nxn image , mxm mask
  float *image = new float[n * n];
  float *mask = new float[m * m];
  float *output = new float[n * n]; // output
  float *output_golden = new float[n * n]; // output

  float min_image, max_image, min_mask, max_mask; // mask, image data ranges
  min_mask = -1.0;
  max_mask = 1.0;
  min_image = -10.0;
  max_image = 10.0;
  std::uniform_real_distribution<float> dist_image(min_image, max_image);
  std::uniform_real_distribution<float> dist_mask(min_mask, max_mask);

  for (unsigned int i = 0; i < n * n; i++) {
    image[i] = dist_image(rand_gen); // here dist is a  callable object
  }

  for (unsigned int i = 0; i < m * m; i++) {
    mask[i] = dist_mask(rand_gen); // here dist is a  callable object
  }
  
  //call mmul with timing measurement
  //double start = omp_get_wtime();
  start = high_resolution_clock::now();
  convolve(image, output, n, mask, m);
  end = high_resolution_clock::now();
  //double end = omp_get_wtime();

  // Convert the calculated duration to a double using the standard library
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  convolution_golden(image, output_golden, (int)n, mask, (int)m);
  //first element 
  //printf("%f\n",output[0]);
  //last element
  //printf("%f\n",output[n*n-1]);
  //time in milliseconds
  //printf("%.10f\n",(end-start)*1000);
  //printf("diff  = %.16g\n", end - start);
  cout << duration_sec.count() << endl;

  compare_matrix(output, output_golden, n);
  //debug prints
  /*print_matrix(image,n);
  print_matrix(mask,m);
  print_matrix(output,n);
  print_matrix(output_golden,n);*/

  delete [] image;
  image = nullptr;
  delete [] mask;
  mask = nullptr;
  delete [] output;
  output = nullptr;
  delete [] output_golden;
  output_golden = nullptr;

return 0;
}
