"""
python main.py --model_name test --n_estimators 2 --max_depth 10 --max_samples 0.2 --sklearn True --SIZE 10
python main.py --model_name test --n_estimators 2 --max_depth 2 --max_samples 0.2 --sklearn False --SIZE 2

python main.py --model_name test --n_estimators 100 --max_depth 100 --max_samples 1.0 --sklearn False --SIZE 10
"""

import numpy as np
import pandas as pd
from collections import Counter
import argparse
import pickle
import os

# just for comparision
from sklearn.ensemble import RandomForestClassifier as sklearn_RandomForestClassifier
# import sklearn
# print(sklearn.ensemble)

from utils import DATA_PATH, DIR_PATH_FIGURES, DIR_PATH_SUBMISSIONS, H, W
from utils import predict_nontest, predict_test, save_for_submission, eval
from data import  get_dataset
from model import RandomForestClassifier



FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Run an experiment competition")
    parser.add_argument("--model_name", type=str, help="to save the sumbssion")
    parser.add_argument("--n_estimators", type=int, help="Number of trees")
    parser.add_argument("--max_depth", type=int, help="Maximum deph")
    parser.add_argument("--criterion", type=str, default="gini", help="criterion")
    parser.add_argument("--max_samples", type=str, default="1.0", help="max samples")    
    parser.add_argument("--sklearn", type=bool_flag, default=False, help="Try sklearn or not")
    parser.add_argument("--SIZE", type=int, default=H, help="HEIGHT, WIDTH : 10, 5, 2 ...")
    params = parser.parse_args()
    print(params)

    if params.SIZE == H :
        HEIGHT, WIDTH = None, None
    else : 
        HEIGHT, WIDTH = params.SIZE, params.SIZE

    # Data
    # Normal training data
    IDs_test, (X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d) = get_dataset(
        train_pct=70, holdout_pct=10, k_fold=False, HEIGHT=HEIGHT, WIDTH=WIDTH, do_over_sampling=False, do_under_sampling=False,
         scaler_class=None, is_pytorch=False, seed=0
    )
    # Training data for k-fold cross validation
    # IDs_test, kfold_iterator = get_dataset(
    #     train_pct=70, holdout_pct=10, k_fold=True, HEIGHT=None, WIDTH=None, do_over_sampling=False, do_under_sampling=False,
    #     scaler_class=None, is_pytorch=False, seed=0
    # )

    if params.sklearn : class_model = sklearn_RandomForestClassifier
    else : class_model = RandomForestClassifier

    params.max_samples = float(params.max_samples)
    if params.max_samples > 1 : params.max_samples = int(params.max_samples)  
    if params.max_samples == 1.0 : params.max_samples = None  
    forest = class_model(
                    n_estimators=params.n_estimators, 
                    criterion=params.criterion,
                    max_depth=params.max_depth, 
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.0,
                    bootstrap=True,
                    random_state=None,
                    verbose=0,
                    max_samples=params.max_samples,
            )

    forest.train(X_tr, Y_tr)

    # is sklearn
    train_acc = eval(forest, X_tr, Y_tr)  # Retrieve train accuracy
    val_acc = eval(forest, X_val, Y_val)  # Retrieve test accuracy
    node_count = 0
    if not params.sklearn : node_count = forest.node_count()

    print('\nMetrics :\n')
    print("train acc : {:7.4f} % \nvalidation acc : {:7.4f} % \nnumber of nodes  : {:7}".format(
            train_acc * 100,
            val_acc * 100,
            node_count
        ))
    
    test_acc = predict_nontest(forest, X_ht_test, Y_ht_test)
    print(test_acc)


    fileName = params.model_name 
    i = 1
    while os.path.exists(f"{DIR_PATH_SUBMISSIONS}/{fileName}_{i}.csv") : i+=1
    fileName = f'{fileName}_{i}'
    print(fileName)

    Y_hat_A, Y_hat_B = predict_test(forest, X_test)
    save_for_submission(IDs_test, Y_hat_A, Y_hat_B, fileName=f"{fileName}.csv")

    with open(f"{DIR_PATH_FIGURES}/{fileName}.pickle","wb") as file_handle:
        to_save = {"model" : forest, "perfs" : [train_acc, val_acc, test_acc]}
        pickle.dump(to_save, file_handle, pickle.HIGHEST_PROTOCOL)