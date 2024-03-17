#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J LSTM_cpp
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH -c 1
g++ LSTM_cpu.cpp LSTM.cpp LSTM_main.cpp -Wall -O3 -std=c++17 -o LSTM_cpp.o
./LSTM_cpp.o 1000 cpu 32 cpp
