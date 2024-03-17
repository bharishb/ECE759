#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:10:00
#SBATCH --nodes=1 --cpus-per-task=10
g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
for j in {1..10}
do
for i in {1..10}
do
	./task2 $((10**6)) $((i))
done
done


for i in {0..99}
do
        if [ $i -eq 0 ]; then
                sed -n "$((2*i+2))p" task2.out > task2c_time_simd.log
        else
                sed -n "$((2*i+2))p" task2.out >> task2c_time_simd.log
        fi
done
