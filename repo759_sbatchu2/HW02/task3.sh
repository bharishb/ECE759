#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:20:00
#SBATCH -c 1
g++ matmul.cpp task3.cpp -Wall -O3 -std=c++17 -o task3
./task3
