#!/bin/bash

# $1 : {all/multi}_{fixd/grow}, $2 : gpu , $3 : approach, $4 : result_dir, $5 : ntasks

if [ "$2" != "" ]; then
  echo "Running on gpu: $2"
else
  echo "No gpu has been assigned."
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

if [ "$3" == "bal_ft" ] || [ "$3" == "bal_lwf" ] || [ "$3" == "bal_joint" ]; then
  echo "Running for approach: $3"
else
  echo "No approach has been assigned or that approach is not available for BALANCING."
fi

RESULTS_DIR="$PROJECT_DIR/results"
if [ "$4" != "" ]; then
    RESULTS_DIR=$4
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    if [ "$1" = "all_fixd" ]; then
      if [ "$3" = "bal_joint" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_fixd_gs_${SEED} \
                   --datasets cifar100_icarl --num-tasks $5 --network resnet32 --seed $SEED \
                   --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                   --gridsearch-tasks $5 --gridsearch-config gridsearch_config \
                   --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                   --approach $3 --gpu $2 --num-epochs-ft 10
      else
        for EXEMPLARS in 2000 1000 500 200
        do
            PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_fixd_gs_${EXEMPLARS}_${SEED} \
                   --datasets cifar100_icarl --num-tasks $5 --network resnet32 --seed $SEED \
                   --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                   --gridsearch-tasks $5 --gridsearch-config gridsearch_config \
                   --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                   --approach $3 --gpu $2 --num-epochs-ft 10 \
                   --num-exemplars $EXEMPLARS --exemplar-selection herding
        done
      fi
    
    elif [ "$1" = "multi_fixd" ]; then
      if [ "$3" = "bal_joint" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_fixd_gs_${SEED} \
                 --datasets cifar100_icarl --num-tasks $5 --network resnet32 --seed $SEED \
                 --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                 --gridsearch-tasks $5 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach $3 --gpu $2 --num-epochs-ft 10 --multi-loss
      else
        for EXEMPLARS in 2000 1000 500 200
        do
            PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_fixd_gs_${EXEMPLARS}_${SEED} \
                   --datasets cifar100_icarl --num-tasks $5 --network resnet32 --seed $SEED \
                   --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                   --gridsearch-tasks $5 --gridsearch-config gridsearch_config \
                   --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                   --approach $3 --gpu $2 --num-epochs-ft 10 --multi-loss \
                   --num-exemplars $EXEMPLARS --exemplar-selection herding
        done
      fi
    
    elif [ "$1" = "all_grow" ]; then
      if [ "$3" = "bal_joint" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_grow_gs_${SEED} \
                 --datasets cifar100_icarl --num-tasks $5 --network resnet32 --seed $SEED \
                 --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                 --gridsearch-tasks $5 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach $3 --gpu $2 --num-epochs-ft 10
    
      else
        for EXEMPLARS in 20 10 5 2
        do
            PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_grow_gs_${EXEMPLARS}_${SEED} \
                   --datasets cifar100_icarl --num-tasks $5 --network resnet32 --seed $SEED \
                   --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                   --gridsearch-tasks $5 --gridsearch-config gridsearch_config \
                   --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                   --approach $3 --gpu $2 --num-epochs-ft 10 \
                   --num-exemplars-per-class $EXEMPLARS --exemplar-selection herding
        done
      fi
    
    elif [ "$1" = "multi_grow" ]; then
      if [ "$3" = "bal_joint" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_grow_gs_${SEED} \
                 --datasets cifar100_icarl --num-tasks $5 --network resnet32 --seed $SEED \
                 --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                 --gridsearch-tasks $5 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach $3 --gpu $2 --num-epochs-ft 10 --multi-loss
      else
        for EXEMPLARS in 20 10 5 2
        do
            PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_grow_gs_${EXEMPLARS}_${SEED} \
                   --datasets cifar100_icarl --num-tasks $5 --network resnet32 --seed $SEED \
                   --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
                   --gridsearch-tasks $5 --gridsearch-config gridsearch_config \
                   --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                   --approach $3 --gpu $2 --num-epochs-ft 10 --multi-loss \
                   --num-exemplars-per-class $EXEMPLARS --exemplar-selection herding
        done
      fi
    else
            echo "No scenario provided."
    fi
done
