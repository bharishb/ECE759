#include "msort.h"
#include <cassert>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <ratio>
using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

void print_vector(int* arr, size_t n){

    for(size_t i=0; i<n; i++){

        cout << arr[i]<<" ";
    }
    cout<<endl;
}

void compare_vector(int* arr, int* arr_golden, size_t n){

    for(size_t i=0; i<n; i++){

        assert(arr[i]==arr_golden[i]);
    }
}


int main(int argc, char** argv){

    //read n value to create matrix nxn
    unsigned int n = atoi(argv[1]);
    //read number of threads to create parallel threads in omp
    unsigned int t = atoi(argv[2]);
    omp_set_num_threads(t); // set t number of parallel threads
    //threshold
    size_t threshold = atoi(argv[3]);

     // random number generator
    std::random_device rd;       // obtain a random number
    std::mt19937 rand_gen(rd()); // seed the generator

    // Time measurement
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec; // milli is from ratio library

    //create n element integer array
    int* arr;
    int* arr_golden;
    arr = new int[n];
    arr_golden = new int[n];
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (unsigned int i = 0; i < n; i++) {
    arr[i] = dist(rand_gen); // here dist is a  callable object
    arr_golden[i] = arr[i];
    }
    //print_vector(arr, n);

    //call mmul with timing measurement
    //double start = omp_get_wtime();
    start = high_resolution_clock::now();
    msort(arr, n, threshold);
    end = high_resolution_clock::now();
    //double end = omp_get_wtime();

    //golden sort
    sort(arr_golden, arr_golden+n);
    compare_vector(arr,arr_golden,n);

    // Convert the calculated duration to a double using the standard library
    duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    //first element 
    printf("%d\n",arr[0]);
    //last element
    printf("%d\n",arr[n-1]);
    //time in milliseconds
    //printf("%.10f\n",(end-start)*1000);
    //printf("diff  = %.16g\n", end - start);
    cout << duration_sec.count() << endl;

    //debug prints
    //print_vector(arr, n);
    //print_vector(arr_golden, n);

    delete [] arr;
    arr = nullptr;

    delete [] arr_golden;
    arr_golden = nullptr;

return 0;
}