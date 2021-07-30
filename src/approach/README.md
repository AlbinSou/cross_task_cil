# Approaches
We include common CIL upper- and lower-bounds (Finetuning and Incremental Joint Training) and our proposed
approaches defined in _**On the importance of cross-task features for class-incremental learning**_ 
([arxiv](https://arxiv.org/abs/2106.11930)).

## Finetuning
`--approach finetuning`

Learning approach which learns each task incrementally while not using any data or knowledge from previous tasks. By
default, weights corresponding to the outputs of previous classes are not updated. This can be changed by using
`--all-outputs`. This approach allows the use of exemplars.

## Incremental Joint Training
`--approach joint`

Learning approach which has access to all data from previous tasks and serves as an upperbound baseline. Joint training 
can be combined with Freezing by using `--freeze-after num_task (int)`. However, this option is disabled (default=-1).

## FT BAL
`--approach bal_ft`

## Joint BAL
`--approach bal_joint`
