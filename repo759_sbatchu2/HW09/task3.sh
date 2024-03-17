#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --ntasks-per-node=2
module load mpi/mpich/4.0.2
mpicxx task3.cpp -Wall -O3 -o task3
srun -n 2 task3 10
