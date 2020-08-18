#!/bin/bash -l
 
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-10
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
 
export LANG="en_US.utf8"
export LANGUAGE="en_US:en"
 
# Activate the correct gpu environment
source ${HOME}/.activate_conda
 
cd $HOME/workspace/your-gpu-project
 

test $SGE_TASK_ID -eq 1 && sleep 10 && PYTHONPATH=. python ./bin/your-project-cli.py --hyperparams 1
test $SGE_TASK_ID -eq 2 && sleep 10 && PYTHONPATH=. python ./bin/your-project-cli.py --hyperparams 2
test $SGE_TASK_ID -eq 3 && sleep 10 && PYTHONPATH=. python ./bin/your-project-cli.py --hyperparams 3
test $SGE_TASK_ID -eq 4 && sleep 10 && PYTHONPATH=. python ./bin/your-project-cli.py --hyperparams 4
test $SGE_TASK_ID -eq 5 && sleep 10 && PYTHONPATH=. python ./bin/your-project-cli.py --hyperparams 5
 
# The counter $SGE_TASK_ID will be incremented up to 10; 1 for each job
[..]