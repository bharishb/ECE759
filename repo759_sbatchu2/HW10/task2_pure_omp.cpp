#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <ctime>
#include "reduce.h"
#include <cassert>


using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
	size_t n = atoi(argv[1]);
	size_t t = atoi(argv[2]);

	omp_set_num_threads(t);

	// Create and fill arr
	float* arr = new float[n];

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1, 1);

	for (size_t i = 0; i < n; i++) {
		arr[i] = dist(generator);
	}

    
      // Time measurement
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec; // milli is from ratio library


    start = high_resolution_clock::now();

	float res = reduce(arr, 0, n);
	
    end = high_resolution_clock::now();
	 // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

      cout<<res<<endl;
   //time taken
    cout << duration_sec.count() << endl;
 
	delete[] arr;

	return 0;
}