#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:10:00
#SBATCH --nodes=1 --cpus-per-task=20
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
for i in {1..20}
do
	./task2 1024 $((i))
done
for i in {0..19}
do
        if [ $i -eq 0 ]; then
                sed -n "$((3*i+3))p" task2.out > task2c_time.log
        else
                sed -n "$((3*i+3))p" task2.out >> task2c_time.log
        fi
done
