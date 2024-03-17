#include <iostream>
#include <string>
#include <random>
#include <cstdlib>
#include <chrono>
#include <stdio.h>
#include <string.h>
#include <cassert>
#include "optimize.h"

using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

void reduce_golden(vec* v, data_t* dest)
{
    *dest =0.0;
    for(size_t i=0; i<v->len; i++)
    {
     *dest += v->data[i];

    }

}


int main(int argc, char** argv) {
	size_t n = std::atoi(argv[1]);

    // Initialize vector
    vec* v = new vec(n);
    v->data = new data_t[n];
    
    // Fill vec with random numbers
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(0, 10);

     // Time measurement
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec; // milli is from ratio library


    for (size_t i = 0; i < n; i++) {
        v->data[i] = (data_t)dist(generator);
    }

    data_t result;
    data_t result_golden;

    reduce_golden(v,&result_golden);

    // Call all optimizeX functions

    // optimize1
    start = high_resolution_clock::now();
    optimize1(v, &result);
    end = high_resolution_clock::now();
    cout<<result<<endl;
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    //time taken
    cout << duration_sec.count() << endl;

    // optimize2
    start = high_resolution_clock::now();
    optimize2(v, &result);
    end = high_resolution_clock::now();
    cout<<result<<endl;
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    //time taken
    cout << duration_sec.count() << endl;

    // optimize3
    start = high_resolution_clock::now();
    optimize3(v, &result);
    end = high_resolution_clock::now();
    cout<<result<<endl;
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    //time taken
    cout << duration_sec.count() << endl;

    // optimize4
    start = high_resolution_clock::now();
    optimize4(v, &result);
    end = high_resolution_clock::now();
    cout<<result<<endl;
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    //time taken
    cout << duration_sec.count() << endl;

    // optimize5
    start = high_resolution_clock::now();
    optimize5(v, &result);
    end = high_resolution_clock::now();
    cout<<result<<endl;
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    //time taken
    cout << duration_sec.count() << endl;  

    //cout<<result_golden<<endl;
    //assert(result_golden==result);

    delete v;
	return 0;
}