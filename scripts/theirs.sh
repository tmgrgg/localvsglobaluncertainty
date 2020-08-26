#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-1
#$ -l tmem=16G
#$ -l h_rt=6:00:00
#$ -l gpu=true

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

# Activate the correct gpu environment
source ${HOME}/.activate_conda

cd $HOME/workspace/your-gpu-project


test $SGE_TASK_ID -eq 1 && sleep 10 && PYTHONPATH=. python3 experiments/train/run_swag.py --data_path=$HOME/Results/theirs/data --epochs=300 --dataset=CIFAR10 --save_freq=300 --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test --dir=$HOME/Results/theirs > out_theirs.log 2>&1

# The counter $SGE_TASK_ID will be incremented up to 10; 1 for each job
[..]
