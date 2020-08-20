#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-30
#$ -l tmem=16G
#$ -l h_rt=48:00:00
#$ -l gpu=true

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

# Activate the correct gpu environment
source ${HOME}/.activate_conda

cd $HOME/localvsglobaluncertainty

test $SGE_TASK_ID -eq 1 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=1 --verbose   > out1.log 2>&1

test $SGE_TASK_ID -eq 2 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=2 --verbose   > out2.log 2>&1

test $SGE_TASK_ID -eq 3 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=3 --verbose   > out3.log 2>&1

test $SGE_TASK_ID -eq 4 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=4 --verbose   > out4.log 2>&1

test $SGE_TASK_ID -eq 5 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=5 --verbose   > out5.log 2>&1

test $SGE_TASK_ID -eq 6 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=6 --verbose   > out6.log 2>&1

test $SGE_TASK_ID -eq 7 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=7 --verbose   > out7.log 2>&1

test $SGE_TASK_ID -eq 8 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=8 --verbose   > out8.log 2>&1

test $SGE_TASK_ID -eq 9 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=9 --verbose   > out9.log 2>&1

test $SGE_TASK_ID -eq 10 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=10 --verbose   > out10.log 2>&1

test $SGE_TASK_ID -eq 11 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=11 --verbose   > out11.log 2>&1

test $SGE_TASK_ID -eq 12 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=12 --verbose   > out12.log 2>&1

test $SGE_TASK_ID -eq 13 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=13 --verbose   > out13.log 2>&1

test $SGE_TASK_ID -eq 14 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=14 --verbose   > out14.log 2>&1

test $SGE_TASK_ID -eq 15 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=15 --verbose   > out15.log 2>&1

test $SGE_TASK_ID -eq 16 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=16 --verbose   > out16.log 2>&1

test $SGE_TASK_ID -eq 17 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=17 --verbose   > out17.log 2>&1

test $SGE_TASK_ID -eq 18 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=18 --verbose   > out18.log 2>&1

test $SGE_TASK_ID -eq 19 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=19 --verbose   > out19.log 2>&1

test $SGE_TASK_ID -eq 20 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=20 --verbose   > out20.log 2>&1

test $SGE_TASK_ID -eq 21 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=21 --verbose   > out21.log 2>&1

test $SGE_TASK_ID -eq 22 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=22 --verbose   > out22.log 2>&1

test $SGE_TASK_ID -eq 23 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=23 --verbose   > out23.log 2>&1

test $SGE_TASK_ID -eq 24 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=24 --verbose   > out24.log 2>&1

test $SGE_TASK_ID -eq 25 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=25 --verbose   > out25.log 2>&1

test $SGE_TASK_ID -eq 26 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=26 --verbose   > out26.log 2>&1

test $SGE_TASK_ID -eq 27 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=27 --verbose   > out27.log 2>&1

test $SGE_TASK_ID -eq 28 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=28 --verbose   > out28.log 2>&1

test $SGE_TASK_ID -eq 29 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=29 --verbose   > out29.log 2>&1

test $SGE_TASK_ID -eq 30 && sleep 10 && PYTHONPATH=. !python3 experiments/train_swag_with_seed/experiment.py --dir=$HOME/Results   --name=multiswag30x30_PreResNet164_CIFAR10 --dataset=CIFAR10 --model=PreResNet164 --criterion=CrossEntropyLoss   --batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9   --training_epochs=160 --swag_epochs=140 --sample_rate=1.0 --rank=30 --cuda --seed=30 --verbose   > out30.log 2>&1


# The counter $SGE_TASK_ID will be incremented up to 10; 1 for each job
[..]