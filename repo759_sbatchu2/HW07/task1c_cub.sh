#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task1_cub
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:05:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0 
nvcc task1_cub.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_cub
for i in {10..20}
do
	./task1_cub $((2**i)) 
done
for i in {0..10}
do
        if [ $i -eq 0 ]; then
                sed -n "$((2*i+2))p" task1_cub.out > task1c_cub.log
        else
                sed -n "$((2*i+2))p" task1_cub.out >> task1c_cub.log
        fi
done
