#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:2:00
#SBATCH -c 1
g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2
./task2 100 50 
