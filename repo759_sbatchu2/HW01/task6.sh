#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task6
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH -c 1
g++ task6.cpp -Wall -O3 -std=c++17 -o task6
./task6 6
