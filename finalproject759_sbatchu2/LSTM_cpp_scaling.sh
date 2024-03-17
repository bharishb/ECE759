#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J LSTM_cpp
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:20:00
#SBATCH -c 1
g++ LSTM_cpu.cpp LSTM.cpp LSTM_main.cpp -Wall -O3 -std=c++17 -o LSTM_cpp.o
for i in {10..20}
do
    ./LSTM_cpp.o 1000 cpu 32 cpp $((2**i))
done
for i in {0..10}
do
        if [ $i -eq 0 ]; then
                sed -n "$((2*i+2))p" LSTM_cpp.out > lstm_cpp_scaling_time.log
        else
                sed -n "$((2*i+2))p" LSTM_cpp.out >> lstm_cpp_scaling_time.log
        fi
done
