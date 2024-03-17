#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:30:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2
for i in {0..19}
do
        echo "iteration $i"
        ./task2 $((2**(10+i))) 128 1024
done
for i in {0..19}
do
        if [ $i -eq 0 ]; then
                sed -n "$((3*i+3))p" task2.out > task2_time_t1024.log
        else
                sed -n "$((3*i+3))p" task2.out >> task2_time_t1024.log
        fi

done
