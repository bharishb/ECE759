#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --nodes=2 --cpus-per-task=20 --ntasks-per-node=1
module load mpi/mpich/4.0.2
mpicxx task2.cpp reduce.cpp -Wall -O3 -o task2 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
for i in {1..20}
do
	srun -n 2 --cpu-bind=none ./task2 $((10**7)) $((i))
        if [ $i -eq 1 ]; then
                sed -n "$((2*i))p" task2.out > task2d1_time.log
        else
                sed -n "$((2*i))p" task2.out >> task2d1_time.log
        fi
done

for i in {1..20}
do
g++ task2_pure_omp.cpp reduce.cpp -Wall -O3 -o task2_pure_omp -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
./task2_pure_omp $((10**7)) $((i))
        if [ $i -eq 1 ]; then
                sed -n "$((2*i+40))p" task2.out > task2d1_time_pure_omp.log
        else
                sed -n "$((2*i+40))p" task2.out >> task2d1_time_pure_omp.log
        fi
done
