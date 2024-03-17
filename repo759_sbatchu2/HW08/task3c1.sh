#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:10:00
#SBATCH --nodes=1 --cpus-per-task=20
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
for i in {1..10}
do
	./task3 $((10**6)) 8 $((2**i))
done
for i in {0..9}
do
        if [ $i -eq 0 ]; then
                sed -n "$((3*i+3))p" task3.out > task3c1_time.log
        else
                sed -n "$((3*i+3))p" task3.out >> task3c1_time.log
        fi
done
