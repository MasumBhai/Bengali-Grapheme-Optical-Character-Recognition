#!/bin/bash

conda activate ml_env
# these all are exported via os.environment to the train.py
export IMG_HEIGHT=137 # these height & weight are fixed for my dataset
export IMG_WIDTH=236
export EPOCHS=20 # for the nural networks, loss function to give us minimum result
export TRAIN_BATCH_SIZE=64
export TEST_BATCH_SIZE=64
export MODEL_MEAN="(0.485, 0.456, 0.406)" # here, mean, std values are from pretrained model `resnet34` used
export MODEL_STD="(0.229, 0.224, 0.225)"
export BASE_MODEL="resnet34"
export TRAINING_FOLDS_CSV="../input/train_folds.csv" # created earlier inside `data_reading.ipynb`

# as i used 5-fold, now need to apply cross validation to each folds
export TRAINING_FOLDS="(0,1,2,3)"
export VALIDATION_FOLDS="(4,)"
conda activate ml_env
python3 train.py # to executing these mentioned folds

export TRAINING_FOLDS="(0,1,2,4)"
export VALIDATION_FOLDS="(3,)"
conda activate ml_env
python3 train.py # to executing these mentioned folds

export TRAINING_FOLDS="(0,1,3,4)"
export VALIDATION_FOLDS="(2,)"
conda activate ml_env
python3 train.py # to executing these mentioned folds

export TRAINING_FOLDS="(0,2,3,4)"
export VALIDATION_FOLDS="(1,)"
conda activate ml_env
python3 train.py # to executing these mentioned folds

export TRAINING_FOLDS="(1,2,3,4)"
export VALIDATION_FOLDS="(0,)"
conda activate ml_env
python3 train.py # to executing these mentioned folds