#!/bin/bash
#SBATCH -C clk
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -c 12
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=5G

time python script_3.py

