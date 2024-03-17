#include "convolution.h"
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{

  #pragma omp parallel for
  for (std::size_t index = 0; index < n*n; index++) {
        std::size_t i_i = index%n;
        std::size_t j_i = index/n;
        for (std::size_t j_mask = 0; j_mask < m;j_mask++) // outmost loop j -> x direction, j -> y direction
        {
            for (std::size_t i_mask = 0; i_mask < m; i_mask++) {
                if ((i_mask == 0) && (j_mask == 0)) {
                    output[i_i + j_i * n] = 0; // initialization
                }
                long long int i_index = i_i + i_mask - (m - 1) / 2; // Image index
                long long int j_index = j_i + j_mask - (m - 1) / 2; // Image Index
                bool cond1 = ((i_index >= 0) && (i_index <(long long int)n)); // typecasting to avoid -ve comparisons
                bool cond2 = ((j_index >= 0) && (j_index < (long long int)n));
                // Edges are when one of the conditions met
                // Corners are when both conditions fail
                float image_value = (cond1 ^ cond2) ? 1 : ((cond1 && cond2) ? image[i_index + j_index * n] : 0);
                output[index] = output[index] + mask[i_mask + j_mask * m] * image_value;
            
            }
        }
  }

}