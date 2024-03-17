#include <omp.h>
#include "../LSTM.h"
//A : pXq, B : qXr, C : pXr
void LSTM::mat_mac_cpu_openmp(const float* A, const float* B, float* C, int p, int q, int r){

//printf("Mat Mul : MAC\n");
//print_matrix(A, p, q);
//print_matrix(B, q, r);
#pragma omp parallel for
for(int i=0; i<p; i++) //rows
{
	for(int j=0; j<r; j++)  // columns
	{
		float temp = 0.0;
		for(int k=0; k<q; k++)
		{
			temp = temp + A[i*q + k]*B[k*r + j]; 
		}
		C[i*r + j] += temp;
	}
}
//print_matrix(C, p, r);

}
