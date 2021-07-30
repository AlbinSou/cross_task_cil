# Cross-task class-incremental learning
Implementation of "[On the importance of cross-task features for class-incremental learning](https://drive.google.com/file/d/1Ygg18cKyTFXjBChung4uHs58w2bHYY_X/view)" 
[[Suppl.](https://drive.google.com/file/d/1npoxgZL43FIq0p4coIqxQ0xITgfAMzE0/view)][[arXiv](https://arxiv.org/abs/2106.11930)][[Poster](./docs/poster_ICML_workshop_2021.pdf)]

Accepted at the International Conference on Machine Learning Workshop (ICML-W) on
[Theory and Foundation of Continual Learning](https://sites.google.com/view/cl-theory-icml2021), 2021.

This implementation extends [FACIL (Framework for Analysis of Class-Incremental Learning)](https://github.com/mmasana/FACIL)
by proposing two new approaches: **BAL_FT** and **BAL_JOINT**. The proposed approaches are in `./src/approach/` and can
be used with the FACIL framework by directly adding them to their corresponding approach folder.

## Installation
Clone this github repository:
```
git clone https://github.com/AlbinSou/cross_task_cil.git
cd cross_task_cil
```

<details>
  <summary>Optionally, create an environment to run the code (click to expand).</summary>

  ### Using a conda environment
  Development environment based on Conda distribution. All dependencies are in `environment.yml` file.

  #### Create env
  To create a new environment check out the repository and type: 
  ```
  conda env create --file environment.yml --name crosstask
  ```
  *Notice:* set the appropriate version of your CUDA driver for `cudatoolkit` in `environment.yml`.

  #### Environment activation/deactivation
  ```
  conda activate crosstask
  conda deactivate
  ```

</details>

Set up your data path by modifying `_BASE_DATA_PATH` in `./src/datasets/dataset_config.py`.

To run the basic code:
```
python3 -u src/main_incremental.py
```
More options are explained in [FACIL](https://github.com/mmasana/FACIL). Also, more specific options on approaches,
loggers, datasets and networks are available on the corresponding `README.md` in the
[FACIL](https://github.com/mmasana/FACIL) subfolders.

## Reproduction of the results
We provide scripts to reproduce the specific scenarios proposed in 
_**On the importance of cross-task features for class-incremental learning**_:

* CIFAR-100 (10 tasks) with ResNet-32 with fixed and growing memory

Our provided results are an average of 10 runs. Check out all available in the [scripts](./scripts) folder.

## License
Please check the MIT license that is listed in this repository.

## Cite
```bibtex
@inproceedings{soutif2021importance,
  title={On the importance of cross-task features for class-incremental learning},
  author={Soutif--Cormerais, Albin and Masana, Marc and Van de Weijer, Joost and Twardowski, Bart{\l}omiej},
  booktitle={International Conference on Machine Learning Workshop},
  year={2021}
}
```
