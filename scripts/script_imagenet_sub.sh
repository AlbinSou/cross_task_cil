#!/bin/bash

# $1 GPU, $2 all/multi, $3 results dir

#if [ "$1" != "" ]; then
#    echo "Running approach: $1"
#else
#    echo "No approach has been assigned."
#fi
if [ "$1" != "" ]; then
    echo "Running on gpu: $1"
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

#for SEED in 0 1 2 3 4 5 6 7 8 9
for SEED in 0
do
  if [ "$2" = "all_joint" ]; then
          PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_${SEED} \
                 --datasets imagenet_subset --num-tasks 25 --network resnet18 --seed $SEED \
                 --nepochs 100 --batch-size 256 --results-path $RESULTS_DIR \
                 --gridsearch-tasks 25 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach bal_joint --gpu $1 --save-models
  elif [ "$2" = "multi_joint" ]; then
          PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_${SEED} \
                 --datasets imagenet_subset --num-tasks 25 --network resnet18 --seed $SEED \
                 --nepochs 100 --batch-size 256 --results-path $RESULTS_DIR \
                 --gridsearch-tasks 25 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach bal_joint --gpu $1 --save-models --multi-loss --num-epochs-ft 10
  elif [ "$2" = "all_grow" ]; then
          PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_${SEED} \
                 --datasets imagenet_subset --num-tasks 25 --network resnet18 --seed $SEED \
                 --nepochs 100 --batch-size 256 --results-path $RESULTS_DIR \
                 --gridsearch-tasks 25 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach bal_ft --gpu $1 --save-models --num-exemplars-per-class 20 --num-epochs-ft 25 --exemplar-selection herding --reinit-heads
  elif [ "$2" = "multi_grow" ]; then
          PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_${SEED} \
                 --datasets imagenet_subset --num-tasks 25 --network resnet18 --seed $SEED \
                 --nepochs 100 --batch-size 256 --results-path $RESULTS_DIR \
                 --gridsearch-tasks 25 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach bal_ft --gpu $1 --save-models --multi-loss --num-exemplars-per-class 20 --num-epochs-ft 25 --exemplar-selection herding --reinit-heads
  else
          echo "No scenario provided."
  fi
done
