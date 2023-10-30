# Towards a Better Negative Sampling Strategy for Dynamic Graphs

* All dynamic graph datasets can be downloaded from [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o).
* DGB package is now available on [pip](https://pypi.org/project/dgb/) (*pip install dgb*). Detailed documentations can be found [here](https://complexdata.ml/docs/proj-tg/dgb/start/).

## Introduction

The different edges in the temporal graph have varying effects as negative samples for training. Edges that haven't appeared in previous history are termed as "Easy Negative Samples", while those that have appeared before are called "Hard Negative Samples". Current methods often employ random sampling to feed different types of negative samples into the model without fully utilizing the potential value of Hard Negative Samples in training.



## Running the experiments

### Set up Environment
```{bash}
conda create -n ENS python=3.9
```

then run 
```{bash}
source install.sh
```

#### Datasets and Processing
All dynamic graph datasets can be downloaded from [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o).
Then, they can be located in *"DG_data"* folder.
For conducting any experiments, the required data should be in the **data** folder under each model of interest.
* For example, to train a *TGN* model on *Wikipedia* dataset, we can use the following command to move the edgelist to the right folder:
```{bash}
cp DG_data/wikipedia.csv tgn/data/wikipedia.csv
```

* Then, the edgelist should be pre-processed to have the right format.
Considering the example of *Wikipedia* edgelist, we can use the following command for pre-processing the dataset:
```{bash}
# JODIE, or TGN
cd tgn
python utils/preprocess_data.py --data wikipedia

# TGAT
python tgat/preprocess_data.py --data wikipedia

# CAWN
cd CAW
python preprocess_data.py --data wikipedia
```


### Model Training&Testing
Just running the script
```{bash}
train.sh

test.sh
```


### Acknowledgment
We would like to thank the authors of [TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs), [TGN](https://github.com/twitter-research/tgn), and [CAWN](https://github.com/snap-stanford/CAW) for providing access to their projects' code.



