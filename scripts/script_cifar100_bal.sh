#!/bin/bash

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

if [ "$1" = "all_fixd" ]; then
  if [ "$3" = "bal_joint" ]; then
    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_fixd_gs \
               --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
               --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
               --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
               --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
               --approach $3 --gpu $2 --num-epochs-ft 10

    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_fixd_gs_500 \
               --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
               --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
               --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
               --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
               --approach bal_ft --gpu $2 --num-epochs-ft 10 \
               --num-exemplars 500 --exemplar-selection herding

  else
    for EXEMPLARS in 2000 1000 500 200
    do
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_fixd_${EXEMPLARS} \
               --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
               --nepochs 100 --batch-size 64 --results-path $RESULTS_DIR --save-models \
               --approach $3 --gpu $2 --num-epochs-ft 10 \
               --lr 0.05 --lr-min 5e-5 --lr-factor 3 --lr-patience 10 \
               --momentum 0.9 --weight-decay 2e-4 \
               --num-exemplars $EXEMPLARS --exemplar-selection herding

        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_fixd_gs_${EXEMPLARS} \
               --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
               --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
               --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
               --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
               --approach $3 --gpu $2 --num-epochs-ft 10 \
               --num-exemplars $EXEMPLARS --exemplar-selection herding
    done
  fi

elif [ "$1" = "multi_fixd" ]; then
  if [ "$3" = "bal_joint" ]; then
    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_fixd_gs \
             --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
             --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
             --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
             --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
             --approach $3 --gpu $2 --num-epochs-ft 10 --multi-loss

    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_fixd_gs_500 \
             --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
             --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
             --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
             --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
             --approach bal_ft --gpu $2 --num-epochs-ft 10 --multi-loss \
             --num-exemplars 500 --exemplar-selection herding

  else
    for EXEMPLARS in 2000 1000 500 200
    do
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_fixd_${EXEMPLARS} \
               --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
               --nepochs 100 --batch-size 64 --results-path $RESULTS_DIR --save-models \
               --approach $3 --gpu $2 --num-epochs-ft 10 --multi-loss \
               --lr 0.05 --lr-min 5e-5 --lr-factor 3 --lr-patience 10 \
               --momentum 0.9 --weight-decay 2e-4 \
               --num-exemplars $EXEMPLARS --exemplar-selection herding

        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_fixd_gs_${EXEMPLARS} \
               --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
               --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
               --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
               --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
               --approach $3 --gpu $2 --num-epochs-ft 10 --multi-loss \
               --num-exemplars $EXEMPLARS --exemplar-selection herding
    done
  fi

elif [ "$1" = "all_grow" ]; then
  if [ "$3" = "bal_joint" ]; then
    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_grow_gs \
             --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
             --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
             --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
             --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
             --approach $3 --gpu $2 --num-epochs-ft 10

    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_grow_gs_500 \
             --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
             --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
             --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
             --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
             --approach bal_ft --gpu $2 --num-epochs-ft 10 \
             --num-exemplars-per-class 500 --exemplar-selection herding

  else
    for EXEMPLARS in 20 10 5 2
    do
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_grow_${EXEMPLARS} \
               --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
               --nepochs 100 --batch-size 64 --results-path $RESULTS_DIR --save-models \
               --approach $3 --gpu $2 --num-epochs-ft 10 \
               --lr 0.05 --lr-min 5e-5 --lr-factor 3 --lr-patience 10 \
               --momentum 0.9 --weight-decay 2e-4 \
               --num-exemplars-per-class $EXEMPLARS --exemplar-selection herding

        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name all_grow_gs_${EXEMPLARS} \
               --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
               --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
               --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
               --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
               --approach $3 --gpu $2 --num-epochs-ft 10 \
               --num-exemplars-per-class $EXEMPLARS --exemplar-selection herding
    done
  fi

elif [ "$1" = "multi_grow" ]; then
  if [ "$3" = "bal_joint" ]; then
    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_grow_gs \
             --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
             --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
             --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
             --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
             --approach $3 --gpu $2 --num-epochs-ft 10 --multi-loss

    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_grow_gs_500 \
             --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
             --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
             --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
             --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
             --approach bal_ft --gpu $2 --num-epochs-ft 10 --multi-loss \
             --num-exemplars-per-class 500 --exemplar-selection herding
  else
    for EXEMPLARS in 20 10 5 2
    do
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_grow_${EXEMPLARS} \
               --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
               --nepochs 100 --batch-size 64 --results-path $RESULTS_DIR --save-models \
               --approach $3 --gpu $2 --num-epochs-ft 10 --multi-loss \
               --lr 0.05 --lr-min 5e-5 --lr-factor 3 --lr-patience 10 \
               --momentum 0.9 --weight-decay 2e-4 \
               --num-exemplars-per-class $EXEMPLARS --exemplar-selection herding

        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name multi_grow_gs_${EXEMPLARS} \
               --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed 0 \
               --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR --save-models \
               --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
               --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
               --approach $3 --gpu $2 --num-epochs-ft 10 --multi-loss \
               --num-exemplars-per-class $EXEMPLARS --exemplar-selection herding
    done
  fi

else
        echo "No scenario provided."
fi
