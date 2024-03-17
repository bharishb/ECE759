#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task4
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0
clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_61 -march=native -o task4 convolve.cpp task4.cpp
./task4 4
