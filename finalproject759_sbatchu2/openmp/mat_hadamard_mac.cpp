#include<omp.h>
#include "../LSTM.h"

void LSTM::mat_hadamard_mac_cpu_openmp(const float* A, const float* B, float* C, int p, int q){

//printf("Mat Hadamard Mac\n");
//print_matrix(A, p, q);
//print_matrix(B, p, q);
    #pragma omp parallel for
	for(int i=0; i<p; i++){
		for(int j=0; j<q; j++) {
			C[i*q + j] += A[i*q + j] * B[i*q + j];
		}
	}
//print_matrix(C, p, q);

}
