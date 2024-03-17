#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --nodes=2 --cpus-per-task=20 --ntasks-per-node=1
module load mpi/mpich/4.0.2
g++ task1.cpp optimize.cpp -Wall -O3 -std=c++17 -o task1 -fno-tree-vectorize
./task1 1000000

for i in {0..5}
do
        if [ $i -eq 0 ]; then
                sed -n "$((2*i+2))p" task1.out > task1c_time_float_sum.log
        else
                sed -n "$((2*i+2))p" task1.out >> task1c_time_float_sum.log
        fi
done
g++ task1.cpp optimize.cpp -Wall -O3 -std=c++17 -o task1 -march=native -fopt-info-vec -ffast-math
./task1 1000000
sed -n "$((12))p" task1.out >> task1c_time_float_sum.log
