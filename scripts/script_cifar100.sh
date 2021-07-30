#!/bin/bash

# $1 : {ctf/noctf}_{fixd/grow/joint}, $2 : gpu , $3 : result_dir, $4 : ntasks

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

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    if [ "$1" = "ctf_fixd" ]; then
        for EXEMPLARS in 2000 1000 500 200
        do
            PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name ctf_fixd_gs_${EXEMPLARS}_${SEED} \
                   --datasets cifar100_icarl --num-tasks $4 --network resnet32 --seed $SEED \
                   --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                   --gridsearch-tasks $4 --gridsearch-config gridsearch_config \
                   --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                   --approach bal_ft --gpu $2 --num-epochs-ft 20 \
                   --num-exemplars $EXEMPLARS --exemplar-selection herding
        done
    elif [ "$1" = "noctf_fixd" ]; then
        for EXEMPLARS in 2000 1000 500 200
        do
            PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name noctf_fixd_gs_${EXEMPLARS}_${SEED} \
                   --datasets cifar100_icarl --num-tasks $4 --network resnet32 --seed $SEED \
                   --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                   --gridsearch-tasks $4 --gridsearch-config gridsearch_config \
                   --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                   --approach bal_ft --gpu $2 --num-epochs-ft 20 --multi-loss \
                   --num-exemplars $EXEMPLARS --exemplar-selection herding
        done
    elif [ "$1" = "ctf_grow" ]; then
        for EXEMPLARS in 20 10 5 2
        do
            PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name ctf_grow_gs_${EXEMPLARS}_${SEED} \
                   --datasets cifar100_icarl --num-tasks $4 --network resnet32 --seed $SEED \
                   --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                   --gridsearch-tasks $4 --gridsearch-config gridsearch_config \
                   --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                   --approach bal_ft --gpu $2 --num-epochs-ft 20 \
                   --num-exemplars-per-class $EXEMPLARS --exemplar-selection herding
        done
    elif [ "$1" = "noctf_grow" ]; then
        for EXEMPLARS in 20 10 5 2
        do
            PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name noctf_grow_gs_${EXEMPLARS}_${SEED} \
                   --datasets cifar100_icarl --num-tasks $4 --network resnet32 --seed $SEED \
                   --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                   --gridsearch-tasks $4 --gridsearch-config gridsearch_config \
                   --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                   --approach bal_ft --gpu $2 --num-epochs-ft 20 --multi-loss \
                   --num-exemplars-per-class $EXEMPLARS --exemplar-selection herding
        done
    elif [ "$1" = "noctf_joint" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name noctf_gs_${SEED} \
                 --datasets cifar100_icarl --num-tasks $4 --network resnet32 --seed $SEED \
                 --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                 --gridsearch-tasks $4 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach bal_joint --gpu $2 --num-epochs-ft 20 --multi-loss --reinit-heads
    elif [ "$1" = "ctf_joint" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name ctf_gs_${SEED} \
                 --datasets cifar100_icarl --num-tasks $4 --network resnet32 --seed $SEED \
                 --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                 --gridsearch-tasks $4 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach bal_joint --gpu $2 --num-epochs-ft 20 --reinit-heads
    else
            echo "No scenario provided."
    fi
done
