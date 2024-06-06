## APK-MRL:  An Adaptive Pre-training Framework with Knowledge-enhanced for Molecular Representation Learning ##

This is the official implementation of  ["APK-MR: An Adaptive Pre-training Framework with Knowledge-enhanced for Molecular Representation Learning"](). In this work, we propose a novel self-supervised framework APK-MRL for molecular representation learning. APK-MRL emancipates molecular comparative learning from high data dependence, tedious manual trial-and-error, and absent domain knowledge, which limit the performance and applicability of existing methods.




## Getting Started

### Installation

- Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
- Install rdkit: conda install -y -c conda-forge rdkit
- Or run the following commands to install both pytorch_geometric and rdkit:

```
# create a new environment
$ conda create --name mol python=3.7
$ conda activate mol

# install requirements
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
$ pip install PyYAML
$ conda install -c conda-forge rdkit=2020.09.1.0
$ conda install -c conda-forge tensorboard
$ conda install -c conda-forge nvidia-apex # optional

# clone the source code of APK-MRL
$ git clone https://github.com/lukcats/APK-MRL.git
$ cd APK-MRL
```

### Dataset

You can download the pre-training data and benchmarks used in the paper [here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view?usp=sharing) and extract the zip file under `./data` folder. The data for pre-training can be found in `pubchem-10m-clean.txt`. All the databases for fine-tuning are saved in the folder under the benchmark name. You can also find the benchmarks from [MoleculeNet](https://moleculenet.org/).

### Pre-training

To train the APK-MRL, where the configurations and detailed explaination for each variable can be found in `config.yaml`
```
$ python pretraining.py
```

### Fine-tuning 

To fine-tune the APK-MRL pre-trained model on downstream molecular benchmarks, where the configurations and detailed explaination for each variable can be found in `config_finetune.yaml`
```
$ python finetune.py
```

## Acknowledgement

- PyTorch implementation of MolCLR: [https://github.com/yuyangw/MolCLR](https://github.com/yuyangw/MolCLR)
- Constructed elements knowledge graph ElementKG: [https://github.com/HICAI-ZJU/KANO](https://github.com/HICAI-ZJU/KANO)
- PyTorch implementation of GraphDTA:[https://github.com/thinng/GraphDTA](https://github.com/yuyangw/MolCLR)
- Strategies for Pre-training Graph Neural Networks: [https://github.com/snap-stanford/pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns)

