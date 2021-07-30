# Cross-task class-incremental learning
Following [FACIL](https://github.com/mmasana/FACIL), run the code with:
```
python3 -u src/main_incremental.py
```
followed by general options:

* `--gpu`: index of GPU to run the experiment on (default=0)
* `--results-path`: path where results are stored (default='../results')
* `--exp-name`: experiment name (default=None)
* `--seed`: random seed (default=0)
* `--save-models`: save trained models (default=False)
* `--no-cudnn-deterministic`: disable CUDNN deterministic (default=False)

and specific options for each of the code parts (corresponding to folders):
  
* `--approach`: learning approach used (default='bal_ft') [[more](approach/README.md)]
* `--datasets`: dataset or datasets used (default=['cifar100']) [[more](datasets/README.md)]
* `--network`: network architecture used (default='resnet32') [[more](networks/README.md)]
* `--log`: loggers used (default='disk') [[more](loggers/README.md)]

go to each of their respective readme to see all available options for each of them.

## Metrics
We provide implementation of our proposed new metrics:

* Cumulative accuracy
* Cumulative forgetting

They can be computed using
```
python cumulative_metrics.py <path_to_result_dir>
```
Where `<path_to_result_dir>` is a directory containing the results from a training session. They will then be stored as
numpy arrays inside `<path_to_result_dir>`. Note that the results dir need to contain model history inside the `models/`
folder, this can be done when training using the `--save-models` option.

## Utils
Some utility are available in `utils.py`.
