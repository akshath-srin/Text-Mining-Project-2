# PIXEL and BERT

This repository contains code for **PIXEL** and **BERT** based on the implementation in this repository https://github.com/xplip/pixel/tree/main

For details about PIXEL, please have a look at our paper [Language Modelling with Pixels](https://arxiv.org/abs/2207.06991). Information on how to cite our work can be found at the bottom.


## Setup

You can set up this codebase as follows to get started with using PIXEL models:


1. Clone repo and initialize submodules
```
git clone https://github.com/xplip/pixel.git
cd pixel
git submodule update --init --recursive
```

2. Create a fresh conda environment
```
conda create -n pixel-env python=3.9
conda activate pixel-env
```

3. Install Python packages
```bash
coconda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge pycairo pygobject manimpango
pip install --upgrade pip
pip install -r requirements.txt
pip install ./datasets
pip install -e .
```

4. (Optional) Install Nvidia Apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout  e5f2f67
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

5. Fine-tuning for Chinese Named Entity Recognition
```bash
# Create a folder in which we keep the data
mkdir -p data
# Get and extract the UD data for parsing and POS tagging
wget -qO- https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4758/ud-treebanks-v2.10.tgz | tar xvz -C data

python scripts/training/run_pos.py \
  --model_name_or_path="Team-PIXEL/pixel-base-finetuned-pos-ud-vietnamese-vtb" \
  --data_dir="data/ud-treebanks-v2.10/UD_Vietnamese-VTB" \
  --remove_unused_columns=False \
  --output_dir="sanity_check" \
  --do_eval \
  --max_seq_length=256 \
  --overwrite_cache
```
