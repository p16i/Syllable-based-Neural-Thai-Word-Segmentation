# Development



## Data Preparation

We have split the original BEST-2010 into training and validation sets. We call this the `raw` data. Essentially, one can create such a dataset by concatenating the original dataset into a text file. We can the splits that we made. Please get in touch.

Note to self: the main data directory is `./data/best-syllable-big`.

### 

 

## Running the Project

Please install all necessary packages via `pip install -r requirements.txt`.

### Training

```
python ./scripts/train.py --model-name seq_sy_ch_conv_3lv \
    --model-params "embc:8|embt:8|embs:8|conv:8|l1:6|do:0.1|oc:BI" \
    --data-dir ./data/best-syllable-big \
    --output-dir ./artifacts/model-test \
    --epoch 1 \
    --batch-size 128 \
    --lr 0.001
```

Available models and their configuration can be found in `./attacut/models`.

### Word Segmenting a text file using a trained model

```
python ./scripts/attacut-cli ../docker-thai-tokenizers/data/wisesight-1000/input.txt \
    --model=./artifacts/model-xx
```

### Evaluation

```
python ./scripts/benchmark.py \
    --label ../docker-thai-tokenizers/data/tnhc/tnhc.label \
    --input=../docker-thai-tokenizers/data/tnhc/input_tokenised-deepcut-deepcut.txt
```

```
# this script will run segmentation and benchmarking in one shot.
python ./scripts/eval.py \
    --model <path-to-model> \
    --dataset <dataset>
```

### Hyperparameter Optimization with Random Search

We use a cluster provided by [GWDG](https://www.gwdg.de) for running random search; the system's queue manager uses `Slurm`.

The script below is for submitting `slurm` jobs for each parameter configuation (see `./scripts/hyper-configs`).

```
python ./scripts/hyperopt.py --config=./scripts/hyper-configs/seq_ch_conv_3lv.yaml \
    --N=20 \
    --max-epoch=20
```

## Utility Scripts
- `./scripts/writing`: we have scripts for generating latex tables used in the paper. These scripts are used via `Make` commands.
- `./scripts/data-related`: we have a couple of scripts for
  1. computing number of words and characters for a dataset;
  2. preprocessing the `THNC` dataset.


## Runtime Evaluation

Please see https://github.com/heytitle/tokenization-speed-benchmark.


## Notebooks

| File | Description |
|-----|-------|
|viz-plot-hyperopt-results.ipynb | making plot for expected valiation performance, i.e. Figure 3 |
|x_attacut_captum.ipynb | explaining model decision, i.e. Figure 4 and 5. for expected valiation performance, i.e. Figure 3 |
|extract-syllable-dict.ipynb| as the name suggested |
|convert-raw-to-syllable-and-label.ipynb | convert BEST-2010 raw dataset to the dataset with syllable labels 


## Backup Files

All data files and models are backup 
```
s3://[backup-bucket]/projects/2020-Syllable-based-Neural-Thai-Word-Segmentation
```

## Misc

**Install Torch with GPU on GWDG's cluster**

```
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```
