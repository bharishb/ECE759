#include <stdio.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include "../LSTM.h"
#include <chrono>
#include <random>
#include <cassert>

using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

void load_data(float* data, const char* filename){
   //printf("Loading Data from %s\n",filename); 
   ifstream in(filename);
   int i=0;
   while(!in.eof()) {
      in>>data[i];
      i++;
   }
}

void dump_data(float* data, int length, const char* filename){
   //printf("Dumping Data To %s\n",filename);
   ofstream out(filename);
 if(out.is_open()){	  
   for(int i=0; i<length; i++) {
              out<<data[i];
	      out<<",\n";
	   }
 }
   out.close();
}

void print_data(float* data, int length){
	printf("Printing Data\n");
	for(int i=0; i<length; i++)
		printf("data[%d] = %f\n", i, data[i]);
   
}

int main(int argc, char** argv){

    // Fill vec with random numbers
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(0, 10);

     // Time measurement
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec; // milli is from ratio library

    int input_size = 1;
    int hidden_size = 2;
    int seq_length = 4;
    int num_inputs = 139;
    int batch_size = atoi(argv[1]);
    char* device = argv[2]; // cpu, gpu
    int t = atoi(argv[3]); // 32 : blockDim
    char* device_language = argv[4]; //cpp, cuda, openmp, mpi
    omp_set_num_threads(t);
    int block_dim = 32;
    int N_iter = atoi(argv[5]);
    printf("Running OPENMP Implementation of LSTM. batch_size = %d, device = %s, num_threads = %d, device_language = %s\n", batch_size, device, omp_get_num_threads(), device_language);

    LSTM LSTM_inst(input_size, hidden_size, seq_length, batch_size, "weights.txt", device, block_dim, device_language);

    LSTM_inst.load_weights();
    //LSTM_inst.print_weights();

    float* inputs = new float[seq_length*num_inputs*N_iter];
    float* outputs = new float[num_inputs*N_iter];
    for(int i=0; i<N_iter; i++) {
        load_data(inputs + i*seq_length*num_inputs,"LSTM_inputs.txt");}
    //print_data(inputs, seq_length*num_inputs*input_size);
    
    start = high_resolution_clock::now();
    LSTM_inst.forward_pass(inputs, outputs, num_inputs*N_iter); 
    end = high_resolution_clock::now();
    
    //print_data(outputs, num_inputs*N_iter);
    //dump_data(outputs, num_inputs*N_iter, "outputs.txt");

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    //time taken
    //cout <<"Duration Taken :" << duration_sec.count() << endl;
    cout <<duration_sec.count() << endl;
    return 0;
}
