#include "LSTM.h"

//Dummy File for CPU to avoid compilation errors for other device/prog_model functions

void LSTM::mat_mul_cpu_openmp(const float* A, const float* B, float* C, int p, int q, int r){}  //cpu openmp
void LSTM::mat_mul_cpu_mpi(const float* A, const float* B, float* C, int p, int q, int r){}  //mpi
void LSTM::mat_mul_gpu(const float* A, const float* B, float* C, int p, int q, int r){}    //gpu

void LSTM::mat_mac_cpu_openmp(const float* A, const float* B, float* C, int p, int q, int r){}
void LSTM::mat_mac_cpu_mpi(const float* A, const float* B, float* C, int p, int q, int r){}
void LSTM::mat_mac_gpu(const float* A, const float* B, float* C, int p, int q, int r){}

void LSTM::mat_add_cpu_openmp(const float* A, const float* B, float* C, int p, int q){}
void LSTM::mat_add_cpu_mpi(const float* A, const float* B, float* C, int p, int q){}
void LSTM::mat_add_gpu(const float* A, const float* B, float* C, int p, int q){}

void LSTM::mat_sgm_cpu_openmp(const float* A, float* C, int p, int q){}
void LSTM::mat_sgm_cpu_mpi(const float* A, float* C, int p, int q){}
void LSTM::mat_sgm_gpu(const float* A, float* C, int p, int q){}

void LSTM::mat_tanh_cpu_openmp(const float* A, float* C, int p, int q){}
void LSTM::mat_tanh_cpu_mpi(const float* A, float* C, int p, int q){}
void LSTM::mat_tanh_gpu(const float* A, float* C, int p, int q){}

void LSTM::mat_hadamard_cpu_openmp(const float* A, const float* B, float* C, int p, int q){}
void LSTM::mat_hadamard_cpu_mpi(const float* A, const float* B, float* C, int p, int q){}
void LSTM::mat_hadamard_gpu(const float* A, const float* B, float* C, int p, int q){}

void LSTM::mat_hadamard_mac_cpu_openmp(const float* A, const float* B, float* C, int p, int q){}
void LSTM::mat_hadamard_mac_cpu_mpi(const float* A, const float* B, float* C, int p, int q){}
void LSTM::mat_hadamard_mac_gpu(const float* A, const float* B, float* C, int p, int q){}

void LSTM::move_inputs_to_gpu(const float* x, int num_inputs){}
void LSTM::move_outputs_to_cpu(float* x, int num_inputs){}
void LSTM::move_params_to_gpu(){}
void LSTM::move_gpu_input_data_to_buffer(const float* A, float* C){}
