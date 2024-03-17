#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:20:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
for i in {5..14}
do
        echo "iteration $i"
        ./task1 $((2**(i))) 32
done
for i in {0..9}
do
        if [ $i -eq 0 ]; then
                sed -n "$((3*i+3))p" task1.out > task1_time_t32.log
        else
                sed -n "$((3*i+3))p" task1.out >> task1_time_t32.log
        fi

done
