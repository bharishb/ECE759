#include <stdio.h>
#include <iostream>
#include <fstream>
#include "LSTM.cuh"
#include <cuda.h>

using namespace std;

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
   printf("Dumping Data To %s\n",filename);
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


    int input_size = 1;
    int hidden_size = 2;
    int seq_length = 4;
    int num_inputs = 139;
    //int num_inputs = 1;
    int batch_size = atoi(argv[1]);
    char* device = argv[2]; // cpu, gpu
    int block_dim = atoi(argv[3]); // 32 : blockDim
    char* device_language = argv[4]; //cpp, cuda, openmp, mpi
    int N_iter = 1000;

    printf("Running GPU Implementation of LSTM\n");

    LSTM LSTM_inst(input_size, hidden_size, seq_length, batch_size, "weights.txt", device, block_dim, device_language);

    LSTM_inst.load_weights();
    //LSTM_inst.print_weights();

    float* inputs = new float[seq_length*num_inputs*N_iter];
    float* outputs = new float[num_inputs*N_iter];
    for(int i=0; i<N_iter; i++)
        load_data(inputs + i*seq_length*num_inputs,"LSTM_inputs.txt");
    //print_data(inputs, seq_length*num_inputs);
//LSTM call with timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    LSTM_inst.forward_pass(inputs, outputs, num_inputs*N_iter);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //print_data(outputs, num_inputs);
    //dump_data(outputs, num_inputs*N_iter, "outputs.txt");

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f\n", ms);  
    return 0;
}
