#!/bin/bash
#SBATCH --time=20
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=16

echo "========== START =========="

echo -n "Start time: "
date

python trainer.py

echo -n "End time: "
date

echo "========== FINISH =========="