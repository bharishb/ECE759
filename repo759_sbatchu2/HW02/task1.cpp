#include "scan.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <ratio>
using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
  char *n_string = argv[1]; // First argument 0 is executable
  int n = atoi(n_string);

  // Time measurement
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec; // milli is from ratio library

  // random number generator
  std::random_device rd;       // obtain a random number
  std::mt19937 rand_gen(rd()); // seed the generator

  // float array memory
  float *arr = new float[n];        // input
  float *output_arr = new float[n]; // output
  float min, max;
  min = -1.0;
  max = 1.0;
  std::uniform_real_distribution<float> dist(min, max);
  for (int i = 0; i < n; i++) {
    arr[i] = dist(rand_gen); // here dist is a  callable object
    assert((((arr[i] < max) &&
             (arr[i] >= min)))); // Assert if the condition is false
  }

  // inclusive scan
  //  Measure duration of scan operation
  start = high_resolution_clock::now();
  scan(arr, output_arr, n);
  end = high_resolution_clock::now();

  // Convert the calculated duration to a double using the standard library
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  // Durations are converted to milliseconds - std::chrono::duration_cast
  // cout << "Total time in scan: " << duration_sec.count() << "ms\n";
  cout << duration_sec.count() << endl;

  // First element
  // cout << "First element in output scan array : " << output_arr[0] <<endl;
  cout << output_arr[0] << endl;

  // Last element
  // cout << "Last element in output scan array : " << output_arr[n-1] <<endl;
  cout << output_arr[n - 1] << endl;

  delete[] arr;
  delete[] output_arr;
  arr = nullptr;
  output_arr = nullptr;

  return 0;
}
