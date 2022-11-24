import os
import ast  # https://www.javatpoint.com/python-ast-module
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics

from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTrain
from tqdm import tqdm
from pytorchtools import EarlyStopping

# pne fine thing, i learnt from many tutorials of many machine learnings is that: for training, it's a best practise
# to declare all necessary declarations in environment file, then use that to everywhere
DEVICE = "cuda"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")

IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))
EPOCHS = int(os.environ.get("EPOCHS"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = os.environ.get("BASE_MODEL")


def main():
    model = MODEL_DISPATCHER[BaseModel](pretrained=True)

    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    # data loader for training dataset
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,  # for training purpose, shuffle gives better performance
        num_workers=4
    )

    # now, will do the same thing for testing
    valid_dataset = BengaliDatasetTrain(
        folds=VALIDATION_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # after training & testing, now need optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # lr is for learning rate,
    # in the future, i want to use  different learning rate for different layer

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="min",
                                                           patience=5,
                                                           factor=0.3, verbose=True)

    early_stopping = EarlyStopping(patience=5, verbose=True)

    if torch.cuda.device_count() > 1:
       model = nn.DataParallel(model)


if __name__ == "__main__":
    main()