#include "convolution.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <ratio>
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

void compare_matrix(const float *X, const float *Y, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      assert((X[i * n + j] == Y[i * n + j]));
    }
  }
}

int main(int argc, char **argv) {
  char *n_string = argv[1]; // First argument 0 is executable
  int n = atoi(n_string);

  char *m_string = argv[2]; // First argument 0 is executable
  int m = atoi(m_string);

  // Time measurement
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec; // milli is from ratio library

  // random number generator
  std::random_device rd;       // obtain a random number
  std::mt19937 rand_gen(rd()); // seed the generator

  // float array memory : nxn image , mxm mask
  float *image = new float[n * n];
  float *mask = new float[m * m];
  float *output = new float[n * n]; // output

  float min_image, max_image, min_mask, max_mask; // mask, image data ranges
  min_mask = -1.0;
  max_mask = 1.0;
  min_image = -10.0;
  max_image = 10.0;
  std::uniform_real_distribution<float> dist_image(min_image, max_image);
  std::uniform_real_distribution<float> dist_mask(min_mask, max_mask);

  for (int i = 0; i < n * n; i++) {
    image[i] = dist_image(rand_gen); // here dist is a  callable object

    // assert if array element out of range
    assert((((image[i] < max_image) &&
             (image[i] >= min_image)))); // Assert if the condition is false
  }

  for (int i = 0; i < m * m; i++) {
    mask[i] = dist_mask(rand_gen); // here dist is a  callable object

    // assert if array element out of range
    assert((((mask[i] < max_mask) &&
             (mask[i] >= min_mask)))); // Assert if the condition is false
  }

  // convolution
  //  Measure duration of convolution operation
  start = high_resolution_clock::now();
  convolve(image, output, n, mask, m);
  end = high_resolution_clock::now();

  // Convert the calculated duration to a double using the standard library
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  // Durations are converted to milliseconds - std::chrono::duration_cast
  // cout << "Total time in convolution: " << duration_sec.count() << "ms\n";
  cout << duration_sec.count() << endl;

  // First element
  cout << output[0] << endl;

  // Last element
  cout << output[n * n - 1] << endl;

  delete[] image;
  delete[] mask;
  delete[] output;
  image = nullptr;
  mask = nullptr;
  output = nullptr;

  return 0;
}
