______________________________________________________________________

<div align="center">

# Breast Cancer Detection (for Kaggle)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


</div>

## Description

### Competition Description
Breast cancer is the most commonly diagnosed cancer type for women worldwide. Early diagnosis increases survival rates up to 99.9%. Mammography is the gold standard in breast cancer screening programs, but despite technological advances, high error rates are still reported. Machine learning techniques, and in particular deep learning (DL), have been successfully used to detect and classify breast cancer.

In this competition, your goal is to correctly estimate BI-RADS score for patients with the given Mammography images.

### Evaluation

As mentioned in Dataset Description, the physicians treats patients with BI-RADS 1 and 2 with the same approach in practice. However, the other BI-RADS scores are treated individually; each of them needs special attention. Therefore, in this challenge, we consider BI-RADS 1 and 2 as negative cases. We use a custom F1 metric for the evaluation. This metric accepts BI-RADS 3, 0, 4, and 5 as positive cases.

### Dataset Description

In this challenge, we provide a curated set of mammography images from four well-known public datasets, which are CBIS-DDSM, CDD-CESM, CMMD, and KAU-BCMD. These datasets are open-access and available for non-commercial uses. Radiology experts of Mogram AI re-annotated images by employing BI-RADS standards.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/kaggle-breast-cancer
cd kaggle-breast-cancer

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/kaggle-breast-cancer
cd kaggle-breast-cancer

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=breast.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
