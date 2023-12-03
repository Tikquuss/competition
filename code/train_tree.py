"""
python train_tree.py --model_name test --n_estimators 100 --max_depth 100 --max_samples 1.0 --sklearn False --SIZE 28 --train_pct 90 --holdout_pct 10 --seed 0
"""

import time

import argparse
import pickle
import os
from pathlib import Path

# just for comparision
from sklearn.ensemble import RandomForestClassifier as sklearn_RandomForestClassifier
from model_tree import RandomForestClassifier

from utils import DATA_PATH, DIR_PATH_FIGURES, DIR_PATH_SUBMISSIONS, H, W
from utils import predict_nontest, predict_test, save_for_submission, eval, bool_flag
from data import  get_dataset
from plotter import confusion_matrix, scores, plot_confusion_matrix

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Run an experiment competition")
    parser.add_argument("--model_name", type=str, help="to save the sumbssion")
    parser.add_argument("--n_estimators", type=int, help="Number of trees")
    parser.add_argument("--max_depth", type=int, help="Maximum deph")
    parser.add_argument("--criterion", type=str, default="gini", help="criterion")
    parser.add_argument("--max_samples", type=str, default="1.0", help="max samples")    
    parser.add_argument("--sklearn", type=bool_flag, default=False, help="Try sklearn or not")
    parser.add_argument("--train_pct", type=float, default=90, help="training data percentage") 
    parser.add_argument("--holdout_pct", type=float, default=10, help="test data percentage")
    parser.add_argument("--SIZE", type=int, default=H, help="HEIGHT, WIDTH : 28, 15, 10...")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    params = parser.parse_args()
    print(params)

    # boostrap (leave fo false, since random forest will do it)
    boostrap=False
    max_samples=0.9
    replace=False

    log_dir = Path(DIR_PATH_FIGURES).parent.absolute()
    #log_dir = path_leaf(DIR_PATH_FIGURES)
    fileName, i = params.model_name, 1
    while os.path.isdir(os.path.join(log_dir, f"log_tree_{fileName}_{i}")) : i+=1
    log_dir = os.path.join(log_dir, f"log_tree_{fileName}_{i}")

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
    ########################################################################
    ########################################################################
    # Model 
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


    ########################################################################
    ########################################################################
    ######## Data
    scaler_class=None
    #scaler_class='standard_scaler'
    scaler_class='min_max_scaler'

    # Normal training data
    IDs_test, (X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d) = get_dataset(
        train_pct=params.train_pct, holdout_pct=params.holdout_pct, k_fold=False, HEIGHT=HEIGHT, WIDTH=WIDTH, do_over_sampling=False, do_under_sampling=False,
         scaler_class=scaler_class, is_pytorch=False, seed=params.seed
    )
    # Training data for k-fold cross validation
    # IDs_test, kfold_iterator = get_dataset(
    #     train_pct=params.train_pct, holdout_pct=params.holdout_pct, k_fold=True, HEIGHT=HEIGHT, WIDTH=WIDTH, do_over_sampling=False, do_under_sampling=False,
    #     scaler_class=scaler_class, is_pytorch=False, seed=params.seed
    # )

    ########################################################################
    ########################################################################

    train_start = time.time()
    ########################################################################
    ########################################################################
    if params.train_pct + params.holdout_pct < 100 : # validation data
        if boostrap : X_tr, Y_tr = resample(X_tr, Y_tr, max_samples=max_samples, replace=replace)
        forest.fit(X_tr, Y_tr)
        train_acc = eval(forest, X_tr, Y_tr) 
        val_acc = eval(forest, X_val, Y_val) 
        test_acc = predict_nontest(forest, X_ht_test, Y_ht_test, seed=params.seed)

        conf_matrix_1 = confusion_matrix(Y_tr, forest.predict(X_tr), n_classes=n_classes)
        conf_matrix_2  = confusion_matrix(Y_ht_test, forest.predict(X_ht_test), n_classes=n_classes)

        Y_hat_A, Y_hat_B = predict_test(forest, X_test)
    else : # No validation data
        if boostrap : X_all, Y_all = resample(X_all, Y_all, max_samples=max_samples, replace=replace)
        forest.fit(X_all, Y_all)
        train_acc = eval(forest, X_all, Y_all)  
        val_acc = -1
        test_acc = predict_nontest(forest, X_ht_test_all, Y_ht_test, seed=params.seed)

        conf_matrix_1 = confusion_matrix(Y_all, forest.predict(X_all), n_classes=n_classes)
        conf_matrix_2 = confusion_matrix(Y_ht_test, forest.predict(X_ht_test_all), n_classes=n_classes)

        Y_hat_A, Y_hat_B = predict_test(forest, X_test_all)

    node_count = 0
    if not params.sklearn : node_count = forest.node_count()

    ########################################################################
    ########################################################################
    training_time = time.time() - train_start

    
    print("train acc : {:7.4f} % \nvalidation acc : {:7.4f} % \nnumber of nodes  : {:7}".format(
        train_acc * 100, val_acc * 100, node_count))
    print("test : ", test_acc)
    print("training time: %s"%(training_time))
        
    plot_confusion_matrix(conf_matrix_1, fileName=f"{fileName}_train", dpf=DIR_PATH_FIGURES)
    _ = scores(conf_matrix_1, fileName=f"{fileName}_train", dpf=DIR_PATH_FIGURES)

    plot_confusion_matrix(conf_matrix_2, fileName=f"{fileName}_test", dpf=DIR_PATH_FIGURES)
    _ = scores(conf_matrix_2, fileName=f"{fileName}_test", dpf=DIR_PATH_FIGURES)


    save_for_submission(IDs_test, Y_hat_A, Y_hat_B, fileName=f"{fileName}.csv", dps=DIR_PATH_SUBMISSIONS)

    with open(f"{log_dir}/{fileName}.pickle" ,"wb") as file_handle:
        to_save = {
            "model" : forest,
            "perfs" : [train_acc, val_acc, test_acc], 
            "others" : [conf_matrix_1, conf_matrix_2],
            "Y_hat_test" : [Y_hat_A, Y_hat_B]
        }
        pickle.dump(to_save, file_handle, pickle.HIGHEST_PROTOCOL)