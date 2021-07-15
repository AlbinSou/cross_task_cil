#!/bin/bash

# $1 approach; $2 gpu; $3 Results; $

if [ "$1" != "" ]; then
    echo "Running approach: $1"
else
    echo "No approach has been assigned."
fi
if [ "$2" != "" ]; then
    echo "Running on gpu: $2"
else
    echo "No gpu has been assigned."
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

RESULTS_DIR="$PROJECT_DIR/results"
if [ "$3" != "" ]; then
    RESULTS_DIR=$3
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"

SEED=0
mem=20
#PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name ${mem}_${SEED} \
# --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed $SEED \
# --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR \
# --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
# --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
# --approach $1 --gpu $2 \
# --num-exemplars-per-class ${mem} --exemplar-selection herding

PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name ${mem}_${SEED} \
 --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed $SEED \
 --nepochs 100 --batch-size 64 --momentum 0.9 --results-path $RESULTS_DIR \
 --approach $1 --gpu $2 \
 --num-exemplars-per-class ${mem} --exemplar-selection herding --lr 0.05 --lr-patience 10 --lr-factor 3 --num-epochs-ft 20
