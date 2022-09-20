## Consistency Regularization for Domain Adaptation

**[[Arxiv]](https://arxiv.org/abs/2208.11084)**

Official GitHub repository for "Consistency Regularization for Domain Adaptation" accepted to OOD-CV 2022 Workshop.

Repository heavily based on [DAFormer's GitHub repository](https://github.com/lhoyer/DAFormer). Please refer to their repository for a more detailed Readme.

We would like to thanks DAFormer for their open source project.

## Setup Environment

We follow DAFormer's environment setup:

```shell
python -m venv ~/venv/daformer
source ~/venv/daformer/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

All experiments were executed on a NVIDIA GeForce RTX 3090.

## Setup Datasets

**Cityscapes:** Download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `${CITYSCAPES_PATH}`.

**GTA:** Download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `${GTA_PATH}`.

**Synthia:** Download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `${SYNTHIA_PATH}`.

Then update `data_root` fields in [uda_gta_to_cityscapes_512x512.py](configs\_base_\datasets\uda_gta_to_cityscapes_512x512.py) and [uda_synthia_to_cityscapes_512x512.py](configs\_base_\datasets\uda_synthia_to_cityscapes_512x512.py) with your `${CITYSCAPES_PATH}`, `${GTA_PATH}` and `${SYNTHIA_PATH}`.

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py ${GTA_PATH} --nproc 8
python tools/convert_datasets/cityscapes.py ${CITYSCAPES_PATH} --nproc 8
python tools/convert_datasets/synthia.py ${SYNTHIA_PATH} --nproc 8
```

## Training

For the experiments in our paper, we use DAFormer's pre-defined configs:

```shell
python run_experiments.py --exp 7
```

More information about DAFormer's available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.

### Consistency Regularization for Domain Adaptation

Code implementation for consistency regularization can be found in [mmseg\models\uda\dacs.py](mmseg\models\uda\dacs.py), mostly under `DACS.forward_train`.

Unfortunately, hyperparameters introduced for consistency regularization have been hard coded. To change them, update `n_pair` and `lambda_sc` in `DACS.forward_train`.

## Testing & Predictions

Trained models can be tested after the training has finished using the following shell command:

```shell
sh test.sh path/to/checkpoint_directory
```
