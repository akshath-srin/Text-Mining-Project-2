# PIXEL and BERT for Chinese NER

This repository contains code for **PIXEL** and **BERT** based on the implementation in this repository https://github.com/xplip/pixel/tree/main


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
# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
export DATA_DIR="chinesedata"
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
export SEQ_LEN=196
export BSZ=64
export GRAD_ACCUM=1
export LR=5e-5
export SEED=42
export NUM_STEPS=15000
  
export RUN_NAME="${LANG}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_ner.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --data_dir=${DATA_DIR} \
  --do_train \
  --do_eval \
  --do_predict \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --early_stopping \
  --early_stopping_patience=5 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=100 \
  --run_name=${RUN_NAME} \
  --output_dir=${RUN_NAME} \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=100 \
  --evaluation_strategy=steps \
  --eval_steps=500 \
  --save_strategy=steps \
  --save_steps=500 \
  --save_total_limit=5 \
  --report_to=wandb \
  --log_predictions \
  --load_best_model_at_end=True \
  --metric_for_best_model="eval_f1" \
  --fp16 \
  --half_precision_backend=apex \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
  --seed=${SEED}
```

If everything is configured correctly, you should expect to see results for PIXEL and BERT similar to the following:

## PIXEL Model Performance on Test Data

- **Accuracy:** 96.21%
- **F1 Score:** 92.28%
- **Precision:** 90.7%
- **Recall:** 93.93%
- **Loss:** 0.2409

## BERT Model Performance on Test Data

- **Accuracy Score:** 88.21%
- **F1 Score:** 68.56%
- **Precision:** 64.44%
- **Recall:** 73.25%
- **Loss:** 0.5625

