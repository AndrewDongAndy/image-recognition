#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=600
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8


# trains on a GPU

echo "========== START =========="

echo -n "Start time: "
date

module load python/3.8
source ~/projects/def-kdong/ml/tensorflow/bin/activate
python fit.py

echo -n "End time: "
date

echo "========== FINISH =========="
