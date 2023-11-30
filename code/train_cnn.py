import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import copy

from utils import DATA_PATH, DIR_PATH_FIGURES, DIR_PATH_SUBMISSIONS, H, W
from utils import predict_nontest, predict_test, save_for_submission, eval, bool_flag
from data import  get_dataset

from model_cnn import MyCNN, MyCNN_2, ResNet9
from trainer import Trainer

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Run an experiment competition")
    parser.add_argument("--model_name", type=str, help="to save the sumbssion")
    parser.add_argument("--max_samples", type=str, default="1.0", help="max samples")    
    parser.add_argument("--SIZE", type=int, default=H, help="HEIGHT, WIDTH : 10, 5, 2 ...")
    params = parser.parse_args()
    print(params)

    if params.SIZE == H :
        HEIGHT, WIDTH = None, None
    else : 
        HEIGHT, WIDTH = params.SIZE, params.SIZE

    n_classes=24
    learning_rate=0.001
    weight_decay=0.0001

    n_epochs = 5# 20#*2
    batch_size = 512*1

    #model = MyCNN(n_classes=n_classes, dropout_conv=0.1, dropout_fc=0.1).to(DTYPE).to(device)
    #model = MyCNN_2(n_classes=n_classes, dropout_conv=0.1, dropout_fc=0.1).to(DTYPE).to(device)
    model = ResNet9(n_classes=n_classes, dropout_conv=0.0, dropout_fc=0.0).to(DTYPE).to(device)
    print(model)

    optimizer = optim.AdamW(model.parameters(), learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    trainer  = Trainer(model, criterion, optimizer)

    #####
    train_accs, train_losses, val_accs, val_losses, best_acc = trainer.train(
        X_tr, Y_tr, X_val, Y_val, batch_size=batch_size, n_epochs = n_epochs, use_tqdm=True)
    train_acc = eval(trainer, X_tr, Y_tr)  # Retrieve train accuracy
    val_acc = eval(trainer, X_val, Y_val)  # Retrieve test accuracy
    test_acc = predict_nontest(trainer, X_ht_test, Y_ht_test, seed=0)
    #####
    train_accs, train_losses, val_accs, val_losses, best_acc = trainer.train(
        X_all, Y_all, X_val=None, Y_val=None, batch_size=batch_size, n_epochs = n_epochs, use_tqdm=True)
    train_acc = eval(trainer, X_all, Y_all)  
    val_acc = -1
    test_acc = predict_nontest(trainer, X_ht_test_all, Y_ht_test, seed=0)
    #####

    print("train acc : {:7.4f} % \nvalidation acc : {:7.4f}".format(train_acc * 100, val_acc * 100))
    print(test_acc)

    fileName = params.model_name
    i = 1
    while os.path.exists(f"{DIR_PATH_SUBMISSIONS}/{fileName}_{i}.csv") : i+=1
    fileName = f'{fileName}_{i}'
    print(fileName)

    plot_training_curve(n_epochs, train_losses, train_accs, val_losses, val_accs, fileName = None)

    Y_hat_A, Y_hat_B = predict_test(trainer, X_test)
    save_for_submission(IDs_test, Y_hat_A, Y_hat_B, fileName=f"{fileName}.csv")

    to_save = {"trainer" : trainer, "perfs" : [train_acc, val_acc, test_acc]}
    torch.save(to_save, f"{DIR_PATH_FIGURES}/{fileName}.pth")