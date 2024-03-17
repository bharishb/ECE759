#include <omp.h>
#include "../LSTM.h"
//sigmoid A : pXq, C : pXq
void LSTM::mat_sgm_cpu_openmp(const float* A, float* C, int p, int q){

//printf("Mat Sigmoid\n");
//print_matrix(A, p, q);
    #pragma omp parallel for
	for(int i=0; i<p; i++){
		for(int j=0; j<q; j++) {
			C[i*q + j] = 1/(1+exp(-A[i*q + j]));
		}
	}
//print_matrix(C, p, q);

}
