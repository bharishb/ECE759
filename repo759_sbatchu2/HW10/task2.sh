#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --nodes=2 --cpus-per-task=20 --ntasks-per-node=1
module load mpi/mpich/4.0.2
mpicxx task2.cpp reduce.cpp -Wall -O3 -o task2 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
srun -n 2 --cpu-bind=none ./task2 1000 10
