#include "mpi.h"
#include <stdio.h>
#include <iostream>
#include <cassert>

using namespace std;

void print_vector(float* arr, unsigned int n, int rank){

   for(unsigned int i=0; i<n; i++)
      cout<<"rank : "<<rank<<","<<arr[i]<<" ; ";
   cout<<endl;
}


int main(int argc, char** argv){
  //read n value to create matrix nxn
  unsigned int n = atoi(argv[1]);

  //MPI message allocation
  float* message_tx = new float[n];
  float* message_rx = new float[n];

  int rank; // rank of process
  int source;
  int destination;
  
  int tag = 0; //setting tag 0 for all messages
  MPI_Status status;

  double start_time, end_time, elapsed_time_r0, elapsed_time_r1;

  //MPI Init and rank
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // getting process rank
  
  //fill the buffers : rank 0 : ascending, rank 1 : descending
  for(unsigned int i=0; i<n; i++){
      if(rank == 0){
        message_tx[i] = (float)i/n;   // increasing
      } else {
        message_tx[i] = (float)(n-i-1)/n;  // decreasing
      }
  }

//communication between two process : ensured no deadlock happens
  if(rank == 0) {
    destination = 1;
    source = 1;
    start_time = MPI_Wtime();
    MPI_Send(message_tx, n, MPI_FLOAT, destination, tag, MPI_COMM_WORLD);
    MPI_Recv(message_rx, n, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
    end_time = MPI_Wtime();
    elapsed_time_r0 = end_time - start_time;
  } else {
    destination = 0;
    source = 0;
    start_time = MPI_Wtime();
    MPI_Recv(message_rx, n, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
    MPI_Send(message_tx, n, MPI_FLOAT, destination, tag, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time_r1 = end_time - start_time;
  }

 if(rank == 0) {
    destination = 1;
    source = 1;
    MPI_Send(&elapsed_time_r0, 1, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
    MPI_Recv(&elapsed_time_r1, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
  } else {
    destination = 0;
    source = 0;
    MPI_Recv(&elapsed_time_r0, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
    MPI_Send(&elapsed_time_r1, 1, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);   
  }

  MPI_Finalize(); //shut down MPI

  if(rank == 0){

    //cout<<elapsed_time_r0<<endl;
    //cout<<elapsed_time_r1<<endl;
    cout<<(elapsed_time_r0+elapsed_time_r1)*1000<<endl; //milliseconds

  }

  //debug prints
  for(unsigned i=0; i<n; i++){
    assert(message_rx[i] == message_tx[n-i-1]);
  }
  //print_vector(message_tx,n,rank);
  //print_vector(message_rx,n,rank);

  //deallocation
  delete [] message_rx;
  message_rx = nullptr;
  delete [] message_tx;
  message_tx = nullptr;

return 0;

}