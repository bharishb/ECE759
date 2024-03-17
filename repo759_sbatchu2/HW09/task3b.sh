#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:10:00
#SBATCH --ntasks-per-node=2
module load mpi/mpich/4.0.2
mpicxx task3.cpp -Wall -O3 -o task3
for i in {1..25}
do
	srun -n 2 task3 $((2**i))
done
for i in {1..25}
do
        if [ $i -eq 1 ]; then
                sed -n "$((i))p" task3.out > task3b_time.log
        else
                sed -n "$((i))p" task3.out >> task3b_time.log
        fi
done
