#!/bin/bash -l
 
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-29
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
 
export LANG="en_US.utf8"
export LANGUAGE="en_US:en"
 
# Activate the correct gpu environment
source ${HOME}/.activate_conda
 
cd $HOME/workspace/your-gpu-project
 
test $SGE_TASK_ID -eq 1 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=2 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 2 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=3 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 3 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=4 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 4 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=5 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 5 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=6 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 6 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=7 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 7 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=8 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 8 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=9 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 9 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=10 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 10 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=11 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 11 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=12 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 12 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=13 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 13 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=14 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 14 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=15 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 15 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=16 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 16 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=17 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 17 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=18 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 18 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=19 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 19 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=20 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 20 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=21 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 21 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=22 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 22 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=23 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 23 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=24 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 24 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=25 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 25 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=26 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 26 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=27 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 27 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=28 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 28 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=29 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

test $SGE_TASK_ID -eq 29 && sleep 10 && PYTHONPATH=. python3 experiments/cache_predictions/experiment.py --dir=$HOME/Results --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss --validation --batch_size=128 --rank=30 --local_samples=30 --cuda --verbose > out_cache_predictions.log 2>&1

# The counter $SGE_TASK_ID will be incremented up to 10; 1 for each job
[..]