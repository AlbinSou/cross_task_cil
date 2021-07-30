# Approaches
We include common CIL upper- and lower-bounds (Finetuning and Incremental Joint Training) and our proposed
approaches defined in _**On the importance of cross-task features for class-incremental learning**_ 
([arxiv](https://arxiv.org/abs/2106.11930)).

## Finetuning
`--approach finetuning`

Learning approach which learns each task incrementally while not using any data or knowledge from previous tasks. By
default, weights corresponding to the outputs of previous classes are not updated. This can be changed by using
`--all-outputs`. This approach allows the use of exemplars.

## FT BAL
`--approach bal_ft`

Learning approach that implements both baselines from the paper. 

The following options are of importance

* `--multi-loss` When used, activates the multi-task loss (version of the baseline that does not learn the cross-task features)
* `--num-exemplars-per-class` Precise the number of exemplars per class to be used (growing memory) (default=20)
* `--num-exemplars` Precise the total number of exemplars to be used (fixed memory)
* `--num-epochs-ft` Precise the number of finetuning epochs to be performed (default=10)


## Joint BAL
`--approach bal_joint`

Learning approach that serves as an upper bound for the baselines, this is equivalent to using the baselines with the maximum amount of memory.

* `--multi-loss` When used, activates the multi-task loss (version of the baseline that does not learn the cross-task features)
* `--num-epochs-ft` Precise the number of finetuning epochs to be performed (default=10)
