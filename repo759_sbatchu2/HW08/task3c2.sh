#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:10:00
#SBATCH --nodes=1 --cpus-per-task=20
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
for i in {1..20}
do
	./task3 $((10**6)) $((i)) $((2**10))
done
for i in {0..19}
do
        if [ $i -eq 0 ]; then
                sed -n "$((3*i+3))p" task3.out > task3c2_time.log
        else
                sed -n "$((3*i+3))p" task3.out >> task3c2_time.log
        fi
done
