#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J LSTM_openmp
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:20:00
#SBATCH --nodes=1 --cpus-per-task=10
g++ LSTM_cpu_openmp.cpp mat_mul.cpp mat_mul_mac.cpp mat_hadamard.cpp mat_hadamard_mac.cpp mat_sgm.cpp mat_tanh.cpp mat_add.cpp ../LSTM.cpp LSTM_main.cpp -Wall -O3 -std=c++17 -o LSTM_openmp.o -fopenmp
./LSTM_openmp.o 1000 cpu 10 openmp
for i in {3..10}
do
	./LSTM_openmp.o 1000 cpu $((i)) openmp $((2**15))
done
for i in {0..10}
do
        if [ $i -eq 0 ]; then
                sed -n "$((2*i+2))p" LSTM_openmp.out > lstm_openmp_scaling_time_thread.log
        else
                sed -n "$((2*i+2))p" LSTM_openmp.out >> lstm_openmp_scaling_time_thread.log
        fi
done
