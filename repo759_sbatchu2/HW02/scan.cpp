#include "scan.h"

// Performs an inclusive scan on input array arr and stores
// the result in the output array
// arr and output are arrays of n elements
// inclusive scan : cumulation/running sum : {a0, a0+a1, a0+a1+a2, ... Sigma(i=0
// to n-1)}
void scan(const float *arr, float *output, std::size_t n) {
  for (std::size_t i = 0; i < n; i++) {
    if (i == 0)
      output[i] = arr[i]; // handling first element seperately since there is no
                          // previous sum
    else
      output[i] = output[i - 1] + arr[i]; // cumulative sum
  }
}
