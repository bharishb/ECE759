#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --nodes=1 --cpus-per-task=20
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp
./task1 4 4
