#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J FirstSlurm
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH -c 2
hostname
