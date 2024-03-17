#include "mpi.h"
#include <iostream>
#include <string>
#include <random>
#include <cstdlib>
#include <chrono>
#include <stdio.h>
#include <string.h>
#include "reduce.h"
#include <cassert>


using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
	int my_rank;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	size_t n = atoi(argv[1]);
	size_t t = atoi(argv[2]);

	omp_set_num_threads(t);

	// Create and fill arr
	float* arr = new float[2 * n];

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1, 1);

	for (size_t i = 0; i < 2 * n; i++) {
		arr[i] = dist(generator);
	}

      // Time measurement
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec; // milli is from ratio library


	// Synchronize processes
	MPI_Barrier(MPI_COMM_WORLD);

    start = high_resolution_clock::now();

	// Call reduce for each MPI process
	float res = reduce(arr, my_rank * n, my_rank * n + n);
	float sum;

	MPI_Reduce(&res, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    end = high_resolution_clock::now();
   // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
 


	if (my_rank == 0) {
    cout<<sum<<endl;
   //time taken
    cout << duration_sec.count() << endl;
    }

	delete[] arr;
	MPI_Finalize();

	return 0;
}