#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3
./task3 10
