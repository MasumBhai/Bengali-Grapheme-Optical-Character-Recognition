# import ast  # https://www.javatpoint.com/python-ast-module
import os

import numpy as np
# import pytorchtools
import sklearn.metrics
import torch
import torch.nn as nn
from datasetClass import BengaliDatasetTrain
from model_dispatcher import MODEL_DISPATCHER
from tqdm import tqdm

# pne fine thing, I learnt from many tutorials of many machine learnings is that: for training, it's the best practise
# to declare all necessary declarations in environment file, then use that to everywhere
DEVICE = "cuda"
IMG_HEIGHT = 137  # these height & weight are fixed for my dataset
IMG_WIDTH = 236
EPOCHS = 20  # for the nural networks, loss function to give us minimum result
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
MODEL_MEAN = [0.485, 0.456, 0.406]  # here, mean, std values are from pretrained model `resnet34` used
MODEL_STD = [0.229, 0.224, 0.225]
BASE_MODEL = "resnet34"
TRAINING_FOLDS_CSV = "../input/train_folds.csv"  # created earlier inside `data_reading.ipynb`

# need to rotate all

TRAINING_FOLDS = [0, 1, 2, 3]
VALIDATION_FOLDS = [4, ]


# TRAINING_FOLDS="(0,1,2,4)"
# VALIDATION_FOLDS="(3,)"
#
# TRAINING_FOLDS="(0,1,3,4)"
# VALIDATION_FOLDS="(2,)"
#
# TRAINING_FOLDS="(0,2,3,4)"
# VALIDATION_FOLDS="(1,)"
#
# TRAINING_FOLDS="(1,2,3,4)"
# VALIDATION_FOLDS="(0,)"


# TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
# 
# IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
# IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))
# EPOCHS = int(os.environ.get("EPOCHS"))
# 
# TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
# TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))
# 
# MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
# MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))
# 
# TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
# VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
# BASE_MODEL = os.environ.get("BASE_MODEL")


def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)

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
        shuffle=True,  # need to check if shuffle value is false, then how my model will react
        num_workers=4
    )

    # after training & testing, now need optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # lr is for learning rate,
    # in the future, I want to use  different learning rate for different layer

    # need to step after every batch & some after every epoch
    # when it's splattering my model scores then I want to reduce the learning rate, taking optimizer as parameter
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="min",  # here, used min mode for loss fuction
                                                           patience=5,
                                                           factor=0.3, verbose=True)

    # to avoid overfitting on the training dataset
    # The EarlyStopping callback can be used to monitor a metric and stop the training when no improvement is observed
    # early_stopping = pytorchtools.EarlyStopping(patience=5, verbose=True)

    # as per my setup, I wanted it to run parallely
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    best_score = -1
    print("FOLD : ", VALIDATION_FOLDS[0])

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_score = train(train_dataset, train_loader, model, optimizer)
        val_loss, val_score = evaluate(valid_dataset, valid_loader, model)

        scheduler.step(val_loss)

        if val_score > best_score:
            best_score = val_score
            # using state_dict for returning a dictionary containing a whole state of the modul
            torch.save(model.state_dict(), f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.pth")

        # just for watching the values, how my model is performing
        epoch_len = len(str(EPOCHS))
        print_msg = (f'[{epoch:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'train_score: {train_score:.5f} ' +
                     f'valid_loss: {val_loss:.5f} ' +
                     f'valid_score: {val_score:.5f}'
                     )
        print(print_msg)

        # my loop breaking condition
        # early_stopping(val_score, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break


def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    # here, used cpu() for copying the tensor to the CPU, but if it is already on the CPU nothing changes
    # then, after that, numpy() creates a NumPy array from the tensor
    y = y.cpu().numpy()

    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    print(
        f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, 'f'total {final_score}, y {y.shape}')

    return final_score


def train(dataset, data_loader, model, optimizer):
    model.train()
    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []

    # bi for batch_index & d is for dataset
    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
        counter = counter + 1
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        # used datatype float, but got some error
        # image = image.to(DEVICE, dtype=torch.long)
        # grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        # vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        # consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        print(image.shape)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        # this is like backward propagation, like discussed in class by adeeb sir
        loss.backward()
        optimizer.step()

        final_loss += loss

        o1, o2, o3 = outputs
        t1, t2, t3 = targets
        final_outputs.append(torch.cat((o1, o2, o3), dim=1))
        final_targets.append(torch.stack((t1, t2, t3), dim=1))

        # if bi % 10 == 0:
        #    break
    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)

    print("=================Train=================")
    macro_recall_score = macro_recall(final_outputs, final_targets)

    return final_loss / counter, macro_recall_score


def evaluate(dataset, data_loader, model):
    with torch.no_grad():
        model.eval()
        final_loss = 0
        counter = 0
        final_outputs = []
        final_targets = []

        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
            counter = counter + 1
            image = d["image"]
            grapheme_root = d["grapheme_root"]
            vowel_diacritic = d["vowel_diacritic"]
            consonant_diacritic = d["consonant_diacritic"]

            # image = image.to(DEVICE, dtype=torch.float)
            # grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
            # vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
            # consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outputs, targets)
            final_loss += loss

            o1, o2, o3 = outputs
            t1, t2, t3 = targets
            # print(t1.shape)
            final_outputs.append(torch.cat((o1, o2, o3), dim=1))
            final_targets.append(torch.stack((t1, t2, t3), dim=1))

        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)

        print("=================Evaluate=================")
        macro_recall_score = macro_recall(final_outputs, final_targets)

    return final_loss / counter, macro_recall_score


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs  # 3 outputs: grapheme, vowel, consonants
    t1, t2, t3 = targets
    layer1 = nn.CrossEntropyLoss()(o1, t1)
    layer2 = nn.CrossEntropyLoss()(o2, t2)
    layer3 = nn.CrossEntropyLoss()(o3, t3)
    return (layer1 + layer2 + layer3) / 3


if __name__ == "__main__":
    main()
