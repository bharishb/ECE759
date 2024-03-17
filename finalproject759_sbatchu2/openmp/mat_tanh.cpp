#include<omp.h>
#include "../LSTM.h"
//tanh A : pXq, C : pXq
void LSTM::mat_tanh_cpu_openmp(const float* A, float* C, int p, int q){

//printf("Mat Tanh\n");
//print_matrix(A, p, q);
    #pragma omp parallel for
	for(int i=0; i<p; i++){
		for(int j=0; j<q; j++) {
			C[i*q + j] = (exp(A[i*q + j]) - exp(-A[i*q + j]))/(exp(A[i*q + j]) + exp(-A[i*q + j]));
		}
	}
//print_matrix(C, p, q);

}

