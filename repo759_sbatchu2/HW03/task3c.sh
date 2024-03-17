#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:10:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3
for i in {0..19}
do
        echo "iteration $i"
        ./task3 $((2**(10+i)))
done
for i in {0..19}
do
        if [ $i -eq 0 ]; then
                sed -n "$((4*i+2))p" task3.out > task3_time_t16.log
        else
                sed -n "$((4*i+2))p" task3.out >> task3_time_t16.log
        fi

done
