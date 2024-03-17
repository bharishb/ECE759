#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:10:00
#SBATCH -c 1
g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1
for i in {0..20}
do
	echo "iteration $i"
	./task1 $((2**(10+i)))
done
for i in {0..20}
do
	if [ $i -eq 0 ]; then
		sed -n "$((4*i+2))p" task1.out > task1_time.log
	else
		sed -n "$((4*i+2))p" task1.out >> task1_time.log
	fi

done
