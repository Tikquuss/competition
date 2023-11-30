"""
python train_cnn.py --model_name test  --learning_rate 0.001 --weight_decay 0.0001 --n_epochs 10 --batch_size 512 --dropout_conv 0.0 --dropout_fc 0.0
"""

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
from plotter import plot_training_curve, confusion_matrix, scores, plot_confusion_matrix, show_example_images
from model_cnn import MyCNN, MyCNN_2, ResNet9
from trainer import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE=torch.float64

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Run an experiment competition")
    parser.add_argument("--model_name", type=str, help="to save the sumbssion")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate") 
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay") 
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs") 
    parser.add_argument("--batch_size", type=int, default=512*1, help="bach size") 
    parser.add_argument("--dropout_conv", type=float, default=0.0, help="dropout convnet") 
    parser.add_argument("--dropout_fc", type=float, default=0.0, help="dropout classifier (fc layer)")     
    parser.add_argument("--SIZE", type=int, default=H, help="HEIGHT, WIDTH : 10, 5, 2 ...")
    params = parser.parse_args()
    print(params)

    if params.SIZE == H : HEIGHT, WIDTH = None, None
    else : HEIGHT, WIDTH = params.SIZE, params.SIZE

    fileName, i = params.model_name, 1
    while os.path.exists(f"{DIR_PATH_SUBMISSIONS}/{fileName}_{i}.csv") : i+=1
    fileName = f'{fileName}_{i}'
    print(fileName)

    n_classes=24
    n_epochs=params.n_epochs
    batch_size=params.batch_size 
    
    #model = MyCNN(n_classes=n_classes, dropout_conv=params.dropout_conv, dropout_fc=params.dropout_fc).to(DTYPE).to(device)
    #model = MyCNN_2(n_classes=n_classes, dropout_conv=params.dropout_conv, dropout_fc=params.dropout_fc).to(DTYPE).to(device)
    model = ResNet9(n_classes=n_classes, dropout_conv=dropout_conv, dropout_fc=params.dropout_fc).to(DTYPE).to(device)
    print(model)

    optimizer = optim.AdamW(model.parameters(), params.learning_rate, weight_decay=params.weight_decay )
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, criterion, optimizer)

    ########################################################################
    ########################################################################

    # train_accs, train_losses, val_accs, val_losses, best_acc = trainer.train(
    #     X_tr, Y_tr, X_val, Y_val, batch_size=batch_size, n_epochs = n_epochs, use_tqdm=True)
    # train_acc = eval(trainer, X_tr, Y_tr) 
    # val_acc = eval(trainer, X_val, Y_val) 
    # test_acc = predict_nontest(trainer, X_ht_test, Y_ht_test, seed=0)

    # conf_matrix_1 = confusion_matrix(Y_tr, trainer.test(X_tr), n_classes=n_classes)
    # conf_matrix_2  = confusion_matrix(Y_ht_test, trainer.test(X_ht_test), n_classes=n_classes)

    # Y_hat_A, Y_hat_B = predict_test(trainer, X_test)

    ########################################################################
    ########################################################################

    train_accs, train_losses, val_accs, val_losses, best_acc = trainer.train(
        X_all, Y_all, X_val=None, Y_val=None, batch_size=batch_size, n_epochs = n_epochs, use_tqdm=True)
    train_acc = eval(trainer, X_all, Y_all)  
    val_acc = -1
    test_acc = predict_nontest(trainer, X_ht_test_all, Y_ht_test, seed=0)

    conf_matrix_1 = confusion_matrix(Y_all, trainer.test(X_all), n_classes=n_classes)
    conf_matrix_2 = confusion_matrix(Y_ht_test, trainer.test(X_ht_test_all), n_classes=n_classes)

    Y_hat_A, Y_hat_B = predict_test(trainer, X_test_all)

    ########################################################################
    ########################################################################

    print("train acc : {:7.4f} % \nvalidation acc : {:7.4f}".format(train_acc * 100, val_acc * 100))
    print(test_acc)

    plot_training_curve(n_epochs, train_losses, train_accs, val_losses, val_accs, fileName = fileName)

    plot_confusion_matrix(conf_matrix_1, fileName=f"{fileName}_train")
    _ = scores(conf_matrix_1, fileName=f"{fileName}_train")

    plot_confusion_matrix(conf_matrix_2, fileName=f"{fileName}_test")
    _ = scores(conf_matrix_2, fileName=f"{fileName}_test")

    save_for_submission(IDs_test, Y_hat_A, Y_hat_B, fileName=f"{fileName}.csv")
    to_save = {
        "trainer" : trainer, 
        "perfs" : [train_acc, val_acc, test_acc], 
        "others" : [conf_matrix_1, conf_matrix_2],
        "Y_hat_test" : [Y_hat_A, Y_hat_B]
    }
    torch.save(to_save, f"{DIR_PATH_FIGURES}/{fileName}.pth")

    