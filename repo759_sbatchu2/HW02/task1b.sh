#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH -c 1
g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1
./task1 1000
