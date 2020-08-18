#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-1
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

# Activate the correct gpu environment
source ${HOME}/.activate_conda

cd $HOME/localvsglobaluncertainty


test $SGE_TASK_ID -eq 1 && sleep 10 && PYTHONPATH=. python3 experiments/train_multiswag/experiment.py --dir=$HOME/Results \
--name=multiswag_heatmap --dataset=FashionMNIST --model=DenseNet10 --criterion=CrossEntropyLoss \
--batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9 \
--training_epochs=150 --swag_epochs=150 --sample_rate=1.0 --rank=90 --num_models=90 --cuda \
> out.log 2>&1

# The counter $SGE_TASK_ID will be incremented up to 10; 1 for each job
[..]