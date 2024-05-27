# Getting Started

## Installation

Set up conda environment and clone the github repo

```sh
# create a new environment
$ conda create --name synthmol python==3.8.10
$ conda activate synthmol

# install requirements
$ pip install torch==2.0.1
$ pip install torch-geometric==1.7.2
$ pip install numpy==1.22.1
$ pip install pandas==2.0.3
$ pip install rdkit==2023.9.4

# clone the source code of SynthMol
$ git clone https://github.com/ThomasSu1/SynthMol.git
$ cd SynthMol/src
```

# Dataset

You can download the dataset file under `SynthMol/Data/` folder. 

# Training

To train the SynthMol model, where the configurations and detailed explanation for each variable can be found in `model_train.py`.

```sh
# Navigate to the source directory
$ cd SynthMol/src

# Run the training script
$ python model_train.py
```
