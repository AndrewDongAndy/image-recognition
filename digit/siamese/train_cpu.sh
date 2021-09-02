#!/bin/bash
#SBATCH --time=360
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=16


# trains on a CPU

echo "========== START =========="

echo -n "Start time: "
date

python fit.py

echo -n "End time: "
date

echo "========== FINISH =========="
