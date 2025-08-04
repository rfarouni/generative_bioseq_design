
<div align="center">

# Project 


</div>

## Description


Uses lightnining hydra for configuration

https://github.com/ashleve/lightning-hydra-template


## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```


## How to run

Train model with default configuration

```bash
# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## Analysis Notebooks

1. [notebooks/1_examine_aav9_model.ipynb](https://github.com/rfarouni/generative_bioseq_design/blob/main/notebooks/1_examine_aav9_model.ipynb)
2. [notebooks/2_classification_with_attention.ipynb](https://github.com/rfarouni/generative_bioseq_design/blob/main/notebooks/2_classification_with_attention.ipynb)
   
   
