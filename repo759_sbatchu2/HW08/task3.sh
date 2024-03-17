#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --nodes=1 --cpus-per-task=20
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
./task3 $((10**6)) 10 3
