#include "cluster.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <ratio>
#include <algorithm>

using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

void cluster_golden(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {

      for(unsigned int i=0; i<n; i++){
        dists[i/(n/t)] +=  std::fabs(arr[i] - centers[i/(n/t)]);
      }

}

void print_vector(float* arr, unsigned int n){
   
   for(unsigned int i=0; i<n; i++)
      cout<<arr[i]<<" ";
   cout<<endl; 
}

void compare_vector(float* arr, float* arr_golden, unsigned int n){
   for(unsigned int i=0; i<n; i++)
      assert(arr[i]==arr_golden[i]);
}

void cluster_false_sharing(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
#pragma omp parallel num_threads(t)
  {
    unsigned int tid = omp_get_thread_num();
#pragma omp for
    for (size_t i = 0; i < n; i++) {
      dists[tid] += std::fabs(arr[i] - centers[tid]);
    }
  }
}


int main(int argc, char** argv){

  //read n value
  unsigned int n = atoi(argv[1]);
  //read number of threads to create parallel threads in omp
  unsigned int t = atoi(argv[2]);
   
  
  // random number generator
  std::random_device rd;       // obtain a random number
  std::mt19937 rand_gen(rd()); // seed the generator

  // Time measurement
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec; // milli is from ratio library

  high_resolution_clock::time_point start_golden;
  high_resolution_clock::time_point end_golden;
  duration<double, std::milli> duration_sec_golden; // milli is from ratio library


  //allocation
  float* arr = new float[n];
  std::uniform_real_distribution<float> dist(0, n);
  for (unsigned int i = 0; i < n; i++) {
    arr[i] = dist(rand_gen); // here dist is a  callable object
  }
  //sort the array
  sort(arr, arr+n);

  float* centers = new float[t];

  for(unsigned int i=1; i<=t; i++){
    centers[i-1] = (2*i-1)*n/(2*t);
  }

  float* dists = new float[t];
  float* dists_golden = new float[t];

  for(unsigned int i=0; i<t; i++){
     dists[i] = 0.0;
     dists_golden[i] = 0.0;
  }

   //cluster function call with timing measurement
   start = high_resolution_clock::now();
   cluster(n,t,arr,centers,dists);
   end = high_resolution_clock::now();
   // Convert the calculated duration to a double using the standard library
   duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);


    //golden reference
    start_golden = high_resolution_clock::now();
    cluster_false_sharing(n,t,arr,centers,dists_golden);
    end_golden = high_resolution_clock::now();
    // Convert the calculated duration to a double using the standard library
    duration_sec_golden = std::chrono::duration_cast<duration<double, std::milli>>(end_golden - start_golden);

   //max distance in the dists array
   float max = 0.0;
   unsigned int thread_id_max = 0;
   for(unsigned int i=0; i<t; i++)
   {
       if(dists[i]>max) {max = dists[i]; thread_id_max = i;}

   }


   //max distance
   printf("%f\n", max);
   //partition ID with max distance
   printf("%d\n",thread_id_max);
   //time taken
   cout << duration_sec.count() << endl;
   //cout << duration_sec_golden.count() << endl;

   //debug prints
   /*print_vector(arr, n);
   print_vector(centers, t);
   print_vector(dists, t);
   print_vector(dists_golden, t);*/
   compare_vector(dists, dists_golden, t);

   //deallocation
   delete [] arr;
   arr = nullptr;
   delete [] dists;
   dists = nullptr;
   delete [] dists_golden;
   dists_golden = nullptr;
   delete [] centers;
   centers = nullptr;

  return 0; 
}
