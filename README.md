# FluorGen

This is the official repository for **FluorGen, a graph-based deep generative model for automated fluorescent dye discovery**. Designing dyes with desired optical properties is challenging due to complex structure–property relationships, but FluorGen addresses this by combining Graph Neural Networks (GNNs), a Variational Autoencoder (VAE), and transfer learning.

The model is pretrained on millions of molecules from the ZINC database, then fine-tuned on FluoDB, a curated dataset of 35,000+ fluorescent compounds. This two-stage approach improves generalization and ensures chemically valid, fluorescence-relevant outputs.

FluorGen supports both unconstrained and scaffold-constrained generation, achieving near-perfect validity (0.999) and high novelty (0.995), surpassing benchmark models. Integrated with KPGT-Fluor property prediction, it enables efficient virtual screening and targeted design of high-performance dyes.

FluorGen provides a scalable and open-source framework to accelerate molecular discovery in bioimaging, diagnostics, and materials science.

![](./flowchart.png)

## Setups

```bash
mamba create -n fluor-gen python=3.10
mamba activate fluor-gen

mamba install -c conda-forge cudatoolkit=11.8.0
python -m pip install nvidia-cudnn-cu11==8.6.0.163 "tensorflow==2.13.*"

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Verify install
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

cd molecule-generation

pip install -e .
pip install rdkit ipykernel loguru matplotlib pandas scikit-learn scipy tqdm dpu-utils more-itertools numpy protobuf
pip install tf2_gnn
```

## Datasets

The full dataset is around 6 GB. You can email us at `molastra@hotmail.com` to request a copy. 😊

## Samples

```bash
num_samples=30000
CUDA_VISIBLE_DEVICES=2 molecule_generation sample checkpoints/MoLeR_checkpoint $num_samples

# Sample from Finetuned Model
CUDA_VISIBLE_DEVICES=6 python scripts/sample.py --model_dir checkpoints/finetune --output_path fluorgen_gen.csv

# Sample from Pretrained Model
CUDA_VISIBLE_DEVICES=7 python scripts/sample.py --model_dir checkpoints/MoLeR_checkpoint --output_path fluorgen_gen.csv

# Sample from Directed Trained Model
CUDA_VISIBLE_DEVICES=7 python scripts/sample.py --model_dir checkpoints/train --output_path fluorgen_train_gen.csv
```

## Encode Fluorecent Molecules

```bash
CUDA_VISIBLE_DEVICES=2 molecule_generation encode MODEL_DIR SMILES_PATH OUTPUT_PATH
```

## Data Preprocessing

```bash
INPUT_DIR=datasets/pretrain/input
OUTPUT_DIR=datasets/pretrain/output
TRACE_DIR=datasets/pretrain/trace

molecule_generation preprocess $INPUT_DIR $OUTPUT_DIR $TRACE_DIR
```

```bash
# Data from FluoDB; Refer ./datasets/build_finetune.ipynb

INPUT_DIR=datasets/finetune/input
OUTPUT_DIR=datasets/finetune/output
TRACE_DIR=datasets/finetune/trace
molecule_generation preprocess $INPUT_DIR $OUTPUT_DIR $TRACE_DIR --pretrained-model-path checkpoints/MoLeR_checkpoint/GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl
```

During finetuning, errors occurred in handling the merge and motif vocabulary. To resolve this, we modified the source code as shown below.

```bash
# Data from FluoDB
## /data/home/silong/projects/fluor/FluorGen/datasets/build_finetune.ipynb
## /data/home/silong/projects/fluor/FluorGen/molecule-generation/molecule_generation/chem/molecule_dataset_utils.py
## Line 671

INPUT_DIR=datasets/finetune/input
OUTPUT_DIR=datasets/finetune/output_ds
TRACE_DIR=datasets/finetune/trace_ds
molecule_generation preprocess $INPUT_DIR $OUTPUT_DIR $TRACE_DIR --pretrained-model-path checkpoints/MoLeR_checkpoint/GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl
```

## Pretrain

`--profile`: run 2 epochs for profiling

```bash
molecule_generation train MoLeR datasets/pretrain/trace \
    --tensorboard
```

## Finetune

```bash
CUDA_VISIBLE_DEVICES=2 molecule_generation train MoLeR datasets/finetune/trace \
    --tensorboard \
    --load-saved-model checkpoints/MoLeR_checkpoint/GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl
```

## Finetune with Peoperty Guided (DyeLeS)

- This setup runs successfully, but the performance is poor. User can explore with fun.

```bash
CUDA_VISIBLE_DEVICES=7 molecule_generation train MoLeR datasets/finetune/trace_ds \
    --tensorboard \
    --load-saved-model checkpoints/MoLeR_checkpoint/GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl

CUDA_VISIBLE_DEVICES=7 molecule_generation train MoLeR datasets/finetune/trace_ds \
    --tensorboard \
    --load-saved-model checkpoints/MoLeR_checkpoint/GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl \
    --load-weights-only \
    --model-params-override '{"num_train_steps_between_valid": 1000}'
```

## Directed Training

```bash
CUDA_VISIBLE_DEVICES=7 molecule_generation train MoLeR datasets/finetune/trace \
    --tensorboard
```

## Others

- [Optimising latent vectors for objective](https://github.com/microsoft/molecule-generation/issues/64)
- [Script for recreating evaluation scores on Guacamol benchmark](https://github.com/microsoft/molecule-generation/issues/43)

## Citations

```bash
<place_holder>
```

This code is adapted from Maziarz’s work, which should also be cited appropriately.

```bash
@article{maziarz2021learning,
  title={Learning to extend molecular scaffolds with structural motifs},
  author={Maziarz, Krzysztof and Jackson-Flux, Henry and Cameron, Pashmina and Sirockin, Finton and Schneider, Nadine and Stiefl, Nikolaus and Segler, Marwin and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:2103.03864},
  year={2021}
}
```

The data used in this study is derived from the works of Zhu, Brown and Polykovskiy, which should be cited appropriately.

```bash
@article{zhuModularArtificialIntelligence2025,
  title = {A Modular Artificial Intelligence Framework to Facilitate Fluorophore Design},
  author = {Zhu, Yuchen and Fang, Jiebin and Ahmed, Shadi Ali Hassen and Zhang, Tao and Zeng, Su and Liao, Jia-Yu and Ma, Zhongjun and Qian, Linghui},
  date = {2025-04-16},
  journaltitle = {Nature Communications},
  shortjournal = {Nature Communications},
  volume = {16},
  number = {1},
  pages = {3598},
  issn = {2041-1723},
  doi = {10.1038/s41467-025-58881-5},
  url = {https://doi.org/10.1038/s41467-025-58881-5}
}
```

```bash
@article{brownGuacaMolBenchmarkingModels2019,
  title = {{{GuacaMol}}: {{Benchmarking Models}} for de {{Novo Molecular Design}}},
  author = {Brown, Nathan and Fiscato, Marco and Segler, Marwin H.S. and Vaucher, Alain C.},
  date = {2019-03-25},
  journaltitle = {Journal of Chemical Information and Modeling},
  shortjournal = {J. Chem. Inf. Model.},
  volume = {59},
  number = {3},
  pages = {1096--1108},
  publisher = {American Chemical Society},
  issn = {1549-9596},
  doi = {10.1021/acs.jcim.8b00839},
  url = {https://doi.org/10.1021/acs.jcim.8b00839}
```

```bash
@article{polykovskiyMolecularSetsMOSES2020,
title = {Molecular Sets ({{MOSES}}): {{A}} Benchmarking Platform for Molecular Generation Models},
author = {Polykovskiy, Daniil and Zhebrak, Alexander and Sanchez-Lengeling, Benjamin and Golovanov, Sergey and Tatanov, Oktai and Belyaev, Stanislav and Kurbanov, Rauf and Artamonov, Aleksey and Aladinskiy, Vladimir and Veselov, Mark and Kadurin, Artur and Johansson, Simon and Chen, Hongming and Nikolenko, Sergey and Aspuru-Guzik, Alán and Zhavoronkov, Alex},
date = {2020},
journaltitle = {Frontiers in Pharmacology},
volume = {11},
issn = {1663-9812},
doi = {10.3389/fphar.2020.565644},
url = {https://www.frontiersin.org/articles/10.3389/fphar.2020.565644}
}
```
