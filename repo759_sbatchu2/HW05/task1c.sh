#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-02:00:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
for i in {2..5}
do
	./task1 $((2**11))  $((2**(i)))
done
for i in {0..3}
do
        if [ $i -eq 0 ]; then
                sed -n "$((9*i+3))p" task1.out > task1_time_m1_2p11.log
        else
                sed -n "$((9*i+3))p" task1.out >> task1_time_m1_2p11.log
        fi

	if [ $i -eq 0 ]; then
                sed -n "$((9*i+6))p" task1.out > task1_time_m2_2p11.log
        else
                sed -n "$((9*i+6))p" task1.out >> task1_time_m2_2p11.log
        fi

	if [ $i -eq 0 ]; then
                sed -n "$((9*i+9))p" task1.out > task1_time_m3_2p11.log
        else
                sed -n "$((9*i+9))p" task1.out >> task1_time_m3_2p11.log
        fi
done
