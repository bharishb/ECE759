#include "convolution.h"
#include <iostream>
using namespace std;

// image is an nxn grid stored in row-major order.
// mask is an mxm grid stored in row-major order.
// Stores the result in output, which is an nxn grid stored in row-major order.
void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) {
  float image_value = 0.0;
  long long int i_index, j_index;
  bool cond1, cond2;
  for (std::size_t j_mask = 0; j_mask < m;
       j_mask++) // outmost loop j -> x direction, j -> y direction
  {
    for (std::size_t i_mask = 0; i_mask < m; i_mask++) {
      for (std::size_t j_i = 0; j_i < n; j_i++) {
        for (std::size_t i_i = 0; i_i < n; i_i++) {
          if ((i_mask == 0) && (j_mask == 0)) {
            output[i_i + j_i * n] = 0; // initialization
          }
          i_index = i_i + i_mask - (m - 1) / 2; // Image index
          j_index = j_i + j_mask - (m - 1) / 2; // Image Index
          cond1 = ((i_index >= 0) &&
                   (i_index <
                    (long long int)n)); // typecasting to avoid -ve comparisons
          cond2 = ((j_index >= 0) && (j_index < (long long int)n));
          // Edges are when one of the conditions met
          // Corners are when both conditions fail
          image_value =
              (cond1 ^ cond2)
                  ? 1
                  : ((cond1 && cond2) ? image[i_index + j_index * n] : 0);
          output[i_i + j_i * n] =
              output[i_i + j_i * n] + mask[i_mask + j_mask * m] * image_value;
        }
      }
    }
  }
}
