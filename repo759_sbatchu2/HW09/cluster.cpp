#include "cluster.h"
#include <cmath>
#include <iostream>

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
#pragma omp parallel num_threads(t)
  {
    unsigned int tid = omp_get_thread_num();
    float dists_thread = 0.0;  // use local or private sum to remove false sharing
#pragma omp for
    for (size_t i = 0; i < n; i++) {
      dists_thread += std::fabs(arr[i] - centers[tid]);
    }
    dists[tid] = dists_thread;  // assigning(dealing with false sharing) at the end after reduction within a thread.
  }
}
