#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:10:00
#SBATCH --nodes=1 --cpus-per-task=20
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp
for i in {1..20}
do
	./task1 1024 $((i))
done
for i in {0..19}
do
        if [ $i -eq 0 ]; then
                sed -n "$((3*i+3))p" task1.out > task1c_time.log
        else
                sed -n "$((3*i+3))p" task1.out >> task1c_time.log
        fi
done
