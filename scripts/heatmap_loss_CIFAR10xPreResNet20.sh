#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-10
#$ -l tmem=16G
#$ -l h_rt=48:00:00
#$ -l gpu=true

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

# Activate the correct gpu environment
source ${HOME}/.activate_conda

cd $HOME/localvsglobaluncertainty

test $SGE_TASK_ID -eq 1 && sleep 10 && PYTHONPATH=. python3 experiments/heatmap_from_models/experiment.py \
--dir=$HOME/Results --name=multiswag30x30_PreResNet20_CIFAR10 --dataset=CIFAR10 --model=PreResNet20 \
--criterion=CrossEntropyLoss --batch_size=128 --min_rank=2 --max_rank=15 --max_num_models=30 \
--local_samples=30 --table_seed=1 --verbose --cuda --on_test  > out_heatmap1a.log 2>&1

test $SGE_TASK_ID -eq 2 && sleep 10 && PYTHONPATH=. python3 experiments/heatmap_from_models/experiment.py \
--dir=$HOME/Results --name=multiswag30x30_PreResNet20_CIFAR10 --dataset=CIFAR10 --model=PreResNet20 \
--criterion=CrossEntropyLoss --batch_size=128 --min_rank=16 --max_rank=30 --max_num_models=30 \
--local_samples=30 --table_seed=1 --verbose --cuda --on_test  > out_heatmap1b.log 2>&1

test $SGE_TASK_ID -eq 3 && sleep 10 && PYTHONPATH=. python3 experiments/heatmap_from_models/experiment.py \
--dir=$HOME/Results --name=multiswag30x30_PreResNet20_CIFAR10 --dataset=CIFAR10 --model=PreResNet20 \
--criterion=CrossEntropyLoss --batch_size=128 --min_rank=2 --max_rank=15 --max_num_models=30 \
--local_samples=30 --table_seed=2 --verbose --cuda --on_test  > out_heatmap2a.log 2>&1


test $SGE_TASK_ID -eq 4 && sleep 10 && PYTHONPATH=. python3 experiments/heatmap_from_models/experiment.py \
--dir=$HOME/Results --name=multiswag30x30_PreResNet20_CIFAR10 --dataset=CIFAR10 --model=PreResNet20 \
--criterion=CrossEntropyLoss --batch_size=128 --min_rank=16 --max_rank=30 --max_num_models=30 \
--local_samples=30 --table_seed=2 --verbose --cuda --on_test  > out_heatmap2b.log 2>&1


test $SGE_TASK_ID -eq 5 && sleep 10 && PYTHONPATH=. python3 experiments/heatmap_from_models/experiment.py \
--dir=$HOME/Results --name=multiswag30x30_PreResNet20_CIFAR10 --dataset=CIFAR10 --model=PreResNet20 \
--criterion=CrossEntropyLoss --batch_size=128 --min_rank=2 --max_rank=15 --max_num_models=30 \
--local_samples=30 --table_seed=3 --verbose --cuda --on_test  > out_heatmap3a.log 2>&1

test $SGE_TASK_ID -eq 6 && sleep 10 && PYTHONPATH=. python3 experiments/heatmap_from_models/experiment.py \
--dir=$HOME/Results --name=multiswag30x30_PreResNet20_CIFAR10 --dataset=CIFAR10 --model=PreResNet20 \
--criterion=CrossEntropyLoss --batch_size=128 --min_rank=16 --max_rank=30 --max_num_models=30 \
--local_samples=30 --table_seed=3 --verbose --cuda --on_test  > out_heatmap3b.log 2>&1

test $SGE_TASK_ID -eq 7 && sleep 10 && PYTHONPATH=. python3 experiments/heatmap_from_models/experiment.py \
--dir=$HOME/Results --name=multiswag30x30_PreResNet20_CIFAR10 --dataset=CIFAR10 --model=PreResNet20 \
--criterion=CrossEntropyLoss --batch_size=128 --min_rank=2 --max_rank=15 --max_num_models=30 \
--local_samples=30 --table_seed=4 --verbose --cuda --on_test  > out_heatmap4a.log 2>&1

test $SGE_TASK_ID -eq 8 && sleep 10 && PYTHONPATH=. python3 experiments/heatmap_from_models/experiment.py \
--dir=$HOME/Results --name=multiswag30x30_PreResNet20_CIFAR10 --dataset=CIFAR10 --model=PreResNet20 \
--criterion=CrossEntropyLoss --batch_size=128 --min_rank=16 --max_rank=30 --max_num_models=30 \
--local_samples=30 --table_seed=4 --verbose --cuda --on_test  > out_heatmap4b.log 2>&1

test $SGE_TASK_ID -eq 9 && sleep 10 && PYTHONPATH=. python3 experiments/heatmap_from_models/experiment.py \
--dir=$HOME/Results --name=multiswag30x30_PreResNet20_CIFAR10 --dataset=CIFAR10 --model=PreResNet20 \
--criterion=CrossEntropyLoss --batch_size=128 --min_rank=2 --max_rank=15 --max_num_models=30 \
--local_samples=30 --table_seed=5 --verbose --cuda --on_test  > out_heatmap5a.log 2>&1

test $SGE_TASK_ID -eq 10 && sleep 10 && PYTHONPATH=. python3 experiments/heatmap_from_models/experiment.py \
--dir=$HOME/Results --name=multiswag30x30_PreResNet20_CIFAR10 --dataset=CIFAR10 --model=PreResNet20 \
--criterion=CrossEntropyLoss --batch_size=128 --min_rank=16 --max_rank=30 --max_num_models=30 \
--local_samples=30 --table_seed=5 --verbose --cuda --on_test  > out_heatmap5b.log 2>&1

# The counter $SGE_TASK_ID will be incremented up to 10; 1 for each job
[..]