#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --nodes=1 --cpus-per-task=20
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
./task2 4 4
