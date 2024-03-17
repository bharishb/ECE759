#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-02:00:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0
nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2
for i in {10..25}
do
        ./task2 $((2**(i))) 32
done
for i in {0..15}
do
        if [ $i -eq 0 ]; then
                sed -n "$((2*i+2))p" task2.out > task2_time_t32.log
        else
                sed -n "$((2*i+2))p" task2.out >> task2_time_t32.log
        fi

done
