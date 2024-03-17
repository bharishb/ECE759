#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0 
nvcc task2.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2
for i in {5..20}
do
	./task2 $((2**i)) 
done
for i in {0..15}
do
        if [ $i -eq 0 ]; then
                sed -n "$((3*i+3))p" task2.out > task2c.log
        else
                sed -n "$((3*i+3))p" task2.out >> task2c.log
        fi
done
