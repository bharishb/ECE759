#include "montecarlo.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <ratio>

using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int montecarlo_golden(const size_t n, const float *x, const float *y, const float radius){


  int count = 0;
  for(size_t i=0; i<n; i++){
    count += (((x[i])*(x[i]) + (y[i])*(y[i])) <= radius*radius);
  }

  return count;
}

void print_vector(float* arr, unsigned int n){

   for(unsigned int i=0; i<n; i++)
      cout<<arr[i]<<" ";
   cout<<endl;
}

int main(int argc, char** argv){

  //read n value
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
  

  high_resolution_clock::time_point start_golden;
  high_resolution_clock::time_point end_golden;
  duration<double, std::milli> duration_sec_golden; // milli is from ratio library



  //allocation
  float radius = 1.0;
  float* x = new float[n];
  float* y = new float[n];
  float* euclidean_distance_square = new float[n];
  std::uniform_real_distribution<float> distx(-radius, radius);  
  std::uniform_real_distribution<float> disty(-radius, radius); 
  for (unsigned int i = 0; i < n; i++) {
    x[i] = distx(rand_gen); // here dist is a  callable object
    y[i] = disty(rand_gen); // here dist is a  callable object
  }

  int count_inside_circle;
  int count_inside_circle_golden;

  //montecarlo function call with timing measurement
   start = high_resolution_clock::now();
   count_inside_circle = montecarlo(n,x,y,radius);
   end = high_resolution_clock::now();
   // Convert the calculated duration to a double using the standard library
   duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

   float estimated_pi = 4*(float)count_inside_circle/n;

   //estimated pi
   cout<<estimated_pi<<endl;
   //time taken
   cout << duration_sec.count() << endl;

   //golden
   start_golden = high_resolution_clock::now();
   count_inside_circle_golden = montecarlo_golden(n,x,y,radius);
   end_golden = high_resolution_clock::now();
   for(unsigned int i=0; i<n; i++) {euclidean_distance_square[i] = ((x[i])*(x[i]) + (y[i])*(y[i]));}
   // Convert the calculated duration to a double using the standard library
   duration_sec_golden = std::chrono::duration_cast<duration<double, std::milli>>(end_golden - start_golden);
   float estimated_pi_golden = 4*(float)count_inside_circle_golden/n;

   //estimated pi golden
   //cout<<estimated_pi_golden<<endl;
   //time taken
   //cout << duration_sec_golden.count() << endl;

   //debug_prints
   /*print_vector(x,n);
   print_vector(y,n);
   print_vector(euclidean_distance_square,n);*/
   assert(estimated_pi == estimated_pi_golden);

   //deallocation
   delete [] x;
   delete [] y;
   delete [] euclidean_distance_square;
   x = nullptr;
   y = nullptr;
   euclidean_distance_square = nullptr;

   return 0;
}
