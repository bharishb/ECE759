#include "LSTM.cuh"

//Dummy File for CPU to avoid compilation errors for other device/prog_model functions

void LSTM::mat_mul_cpu_openmp(const float* A, const float* B, float* C, int p, int q, int r){}  //cpu openmp
void LSTM::mat_mul_cpu_mpi(const float* A, const float* B, float* C, int p, int q, int r){}  //mpi

void LSTM::mat_mac_cpu_openmp(const float* A, const float* B, float* C, int p, int q, int r){}
void LSTM::mat_mac_cpu_mpi(const float* A, const float* B, float* C, int p, int q, int r){}

void LSTM::mat_add_cpu_openmp(const float* A, const float* B, float* C, int p, int q){}
void LSTM::mat_add_cpu_mpi(const float* A, const float* B, float* C, int p, int q){}

void LSTM::mat_sgm_cpu_openmp(const float* A, float* C, int p, int q){}
void LSTM::mat_sgm_cpu_mpi(const float* A, float* C, int p, int q){}

void LSTM::mat_tanh_cpu_openmp(const float* A, float* C, int p, int q){}
void LSTM::mat_tanh_cpu_mpi(const float* A, float* C, int p, int q){}

void LSTM::mat_hadamard_cpu_openmp(const float* A, const float* B, float* C, int p, int q){}
void LSTM::mat_hadamard_cpu_mpi(const float* A, const float* B, float* C, int p, int q){}

void LSTM::mat_hadamard_mac_cpu_openmp(const float* A, const float* B, float* C, int p, int q){}
void LSTM::mat_hadamard_mac_cpu_mpi(const float* A, const float* B, float* C, int p, int q){}
