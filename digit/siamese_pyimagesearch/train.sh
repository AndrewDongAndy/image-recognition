#!/bin/bash
#SBATCH --time=360
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=16

echo "========== START =========="

echo -n "Start time: "
date

python train_siamese_network.py

echo -n "End time: "
date

echo "========== FINISH =========="