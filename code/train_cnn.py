"""
How to use?

python train_cnn.py --model_name test  --learning_rate 0.001 --weight_decay 0.0001 --scheduler False --n_epochs 50 --batch_size 512 --dropout_conv 0.0 --dropout_fc 0.0 --train_pct 90 --holdout_pct 10 --seed 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import copy
import time

import argparse
import os
from pathlib import Path

from utils import DATA_PATH, DIR_PATH_FIGURES, DIR_PATH_SUBMISSIONS, H, W
from utils import predict_nontest, predict_test, save_for_submission, eval, bool_flag, path_leaf
from data import  get_dataset
from plotter import plot_training_curve, confusion_matrix, scores, plot_confusion_matrix, show_example_images
from model_cnn import MyCNN, MyCNN_2, ResNet, Net
from trainer import Trainer, get_dataset_pytorch, resample

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE=torch.float64

if __name__ == "__main__" :

    ########################################################################
    ########################################################################
    # Params
    parser = argparse.ArgumentParser(description="Run an experiment competition")
    parser.add_argument("--model_name", type=str, help="to save the submission")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate") 
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay") 
    parser.add_argument("--scheduler", type=bool_flag, default=False, help="use exponential lr scheduler or not") 
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs") 
    parser.add_argument("--batch_size", type=int, default=512*1, help="bach size") 
    parser.add_argument("--dropout_conv", type=float, default=0.0, help="dropout convnet") 
    parser.add_argument("--dropout_fc", type=float, default=0.0, help="dropout classifier (fc layer)") 
    parser.add_argument("--train_pct", type=float, default=90, help="training data percentage") 
    parser.add_argument("--holdout_pct", type=float, default=10, help="test data percentage") 
    parser.add_argument("--SIZE", type=int, default=H, help="HEIGHT, WIDTH : 10, 5, 2 ... (upsample?, downsample?)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    params = parser.parse_args()
    print(params)

    # boostrap
    boostrap=False
    max_samples=0.9
    replace=False

    ########################################################################
    ########################################################################
    # Where to log the experiments
    log_dir = Path(DIR_PATH_FIGURES).parent.absolute()
    #log_dir = path_leaf(DIR_PATH_FIGURES)
    fileName, i = params.model_name, 1
    while os.path.isdir(os.path.join(log_dir, f"log_{fileName}_{i}")) : i+=1
    log_dir = os.path.join(log_dir, f"log_{fileName}_{i}")

    DIR_PATH_FIGURES = os.path.join(log_dir, "figures")
    DIR_PATH_SUBMISSIONS = os.path.join(log_dir, "submissions")
    os.makedirs(DIR_PATH_SUBMISSIONS, exist_ok=True)
    os.makedirs(DIR_PATH_FIGURES, exist_ok=True)
    print(f"log_dir {log_dir}")
    print(f"DIR_PATH_SUBMISSIONS {DIR_PATH_SUBMISSIONS}")
    
    if params.SIZE == H : HEIGHT, WIDTH = None, None
    else : HEIGHT, WIDTH = params.SIZE, params.SIZE

    fileName, i = params.model_name, 1
    while os.path.exists(f"{DIR_PATH_SUBMISSIONS}/{fileName}_{i}.csv") : i+=1
    fileName = f'{fileName}_{i}'
    print(fileName)

    n_classes=24
    n_epochs=params.n_epochs
    batch_size=params.batch_size 
    
    ########################################################################
    ########################################################################
    # Model & Optimizer & Criterion Trainer

    #model = MyCNN(n_classes=n_classes, dropout_conv=params.dropout_conv, dropout_fc=params.dropout_fc)
    #model = MyCNN_2(n_classes=n_classes, dropout_conv=params.dropout_conv, dropout_fc=params.dropout_fc)
    model = ResNet(n_classes=n_classes, dropout_conv=params.dropout_conv, dropout_fc=params.dropout_fc)
    model.init_fc(dropout_fc=params.dropout_fc, n_classes=n_classes, x=torch.randn(1, 1, HEIGHT, WIDTH))
    #print(model)

    model = model.to(DTYPE).to(device)
    optimizer = optim.AdamW(model.parameters(), params.learning_rate, weight_decay=params.weight_decay)
    if params.scheduler:
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    else :
        lr_scheduler = None
    criterion = nn.CrossEntropyLoss()
    checkpoint_path=f"{log_dir}/checkpoints" 
    trainer = Trainer(model, criterion, optimizer, lr_scheduler, checkpoint_path)

    ########################################################################
    ########################################################################
    ######## Data
    scaler_class=None
    #scaler_class='standard_scaler'
    scaler_class='min_max_scaler'

    train_transforms = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
    test_transforms = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])

    IDs_test, (X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d) = get_dataset_pytorch(
        train_pct=params.train_pct, holdout_pct=params.holdout_pct, train_transforms=train_transforms, test_transforms=test_transforms, 
        k_fold=False, HEIGHT=HEIGHT, WIDTH=WIDTH, do_over_sampling=False, do_under_sampling=False,
        scaler_class=scaler_class, is_pytorch=True, device=device, seed=params.seed
        )

    train_start = time.time()
    ########################################################################
    ########################################################################
    if params.train_pct + params.holdout_pct < 100 : # validation data
        if boostrap : X_tr, Y_tr = resample(X_tr, Y_tr, max_samples=max_samples, replace=replace)
        train_accs, train_losses, val_accs, val_losses, best_acc = trainer.train(
            X_tr, Y_tr, X_val, Y_val, batch_size=batch_size, n_epochs = n_epochs, use_tqdm=True)
        train_acc = eval(trainer, X_tr, Y_tr) 
        val_acc = eval(trainer, X_val, Y_val) 
        test_acc = predict_nontest(trainer, X_ht_test, Y_ht_test, seed=params.seed)

        conf_matrix_1 = confusion_matrix(Y_tr, trainer.test(X_tr), n_classes=n_classes)
        conf_matrix_2  = confusion_matrix(Y_ht_test, trainer.test(X_ht_test), n_classes=n_classes)

        Y_hat_A, Y_hat_B = predict_test(trainer, X_test)
    else : # No validation data
        if boostrap : X_all, Y_all = resample(X_all, Y_all, max_samples=max_samples, replace=replace)
        train_accs, train_losses, val_accs, val_losses, best_acc = trainer.train(
            X_all, Y_all, X_val=None, Y_val=None, batch_size=batch_size, n_epochs = n_epochs, use_tqdm=True)
        train_acc = eval(trainer, X_all, Y_all)  
        val_acc = -1
        test_acc = predict_nontest(trainer, X_ht_test_all, Y_ht_test, seed=params.seed)

        conf_matrix_1 = confusion_matrix(Y_all, trainer.test(X_all), n_classes=n_classes)
        conf_matrix_2 = confusion_matrix(Y_ht_test, trainer.test(X_ht_test_all), n_classes=n_classes)

        Y_hat_A, Y_hat_B = predict_test(trainer, X_test_all)

    ########################################################################
    ########################################################################
    training_time = time.time() - train_start

    print("train acc : {:7.4f} % \nvalidation acc : {:7.4f}".format(train_acc * 100, val_acc * 100))
    print("test : ", test_acc)
    print("training time: %s"%(training_time))

    ########################################################################
    ########################################################################
    # Confusion matrix
    plot_training_curve(n_epochs, train_losses, train_accs, val_losses, val_accs, fileName=fileName, dpf=DIR_PATH_FIGURES)

    plot_confusion_matrix(conf_matrix_1, fileName=f"{fileName}_train", dpf=DIR_PATH_FIGURES)
    _ = scores(conf_matrix_1, fileName=f"{fileName}_train", dpf=DIR_PATH_FIGURES)

    plot_confusion_matrix(conf_matrix_2, fileName=f"{fileName}_test", dpf=DIR_PATH_FIGURES)
    _ = scores(conf_matrix_2, fileName=f"{fileName}_test", dpf=DIR_PATH_FIGURES)

    ########################################################################
    ########################################################################
    # Save the test set prediction is csv format for submission
    save_for_submission(IDs_test, Y_hat_A, Y_hat_B, fileName=f"{fileName}.csv", dps=DIR_PATH_SUBMISSIONS)

    ########################################################################
    ########################################################################
    # Save the model & meta-data
    to_save = {
        "trainer" : trainer, 
        "progress" : [train_accs, train_losses, val_accs, val_losses, best_acc],
        "perfs" : [train_acc, val_acc, test_acc], 
        "others" : [conf_matrix_1, conf_matrix_2],
        "Y_hat_test" : [Y_hat_A, Y_hat_B]
    }
    torch.save(to_save, f"{log_dir}/{fileName}.pth")



    