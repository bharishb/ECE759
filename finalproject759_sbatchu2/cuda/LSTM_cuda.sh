#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J LSTM_cuda
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:10:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0
nvcc LSTM_main.cu LSTM.cu LSTM_gpu.cu gpu_functions.cu move_input_data_to_buffer.cu reset_all_network_states_gpu.cu mat_mul.cu mat_mul_mac.cu mat_hadamard.cu mat_hadamard_mac.cu mat_tanh.cu mat_sgm.cu mat_add.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o LSTM_cuda
./LSTM_cuda 256 gpu 16 cuda
