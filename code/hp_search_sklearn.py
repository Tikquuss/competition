###############################
# Imports from our code
###############################

from utils import DATA_PATH, DIR_PATH_FIGURES, DIR_PATH_SUBMISSIONS, H, W
from utils import predict_nontest, predict_test, save_for_submission, eval
from data import  get_dataset
from plotter import plot_cdf, custom_imshow, confusion_matrix, scores, plot_confusion_matrix, show_example_images

# Import General

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import tqdm
# import itertools
from collections import Counter
import string

import pickle

###############################
# Data
###############################

#HEIGHT, WIDTH = 60, 60
#HEIGHT, WIDTH = 25, 25
# HEIGHT, WIDTH = 10, 10
#HEIGHT, WIDTH = 5, 5
#HEIGHT, WIDTH = 2, 2
HEIGHT, WIDTH = None, None

scaler_class=None
#scaler_class='standard_scaler'
scaler_class='min_max_scaler'

train_pct, holdout_pct = 80, 10
#train_pct, holdout_pct = 90, 10
#train_pct, holdout_pct = 100,0

seed=0
n_classes=24

IDs_test, (X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d) = get_dataset(
    train_pct=train_pct, holdout_pct=holdout_pct, k_fold=False, HEIGHT=HEIGHT, WIDTH=WIDTH, do_over_sampling=False, do_under_sampling=False,
    scaler_class=scaler_class, is_pytorch=False, seed=seed
)

################## Import for sklearn
from sklearn.ensemble import RandomForestClassifier as sklearn_RandomForestClassifier
import itertools
import time
from pathlib import Path

fileName="report_sktree_hpsearch_13_12_(4)"
#fileName="test_test_test"

log_dir = Path(DIR_PATH_FIGURES).parent.absolute()
log_dir = os.path.join(log_dir, f"{fileName}")
DIR_PATH_FIGURES__ = os.path.join(log_dir, "figures")
DIR_PATH_SUBMISSIONS__ = os.path.join(log_dir, "submissions")
os.makedirs(DIR_PATH_SUBMISSIONS__, exist_ok=True)
os.makedirs(DIR_PATH_FIGURES__, exist_ok=True)

""""
m : the number of trees
r : maximum depth
n_prime : number of samples to draw to train each base tree 
d_prime : number of features to consider for each tree  ($d' = \lfloor \sqrt{d} \rfloor$ ...)
f  : purity criteria (Gini index or entropy). 
"""

all_m = np.array([1,  50,  100, 200, 300])
all_r = np.array([10,  100, 150, 200, 300])
all_n_prime = [0.1, 0.5,  1.0]
all_d_prime = [28, 0.5, 1.0] # 28 = sqrt(d)
#all_f = ["gini", "entropy"]
all_f = ["gini"]

all_hparams = itertools.product(all_m, all_r, all_n_prime, all_d_prime, all_f)
all_hparams = list(all_hparams)
len(all_hparams)

all_performances = {}
all_models = {}

for m, r, n_prime, d_prime, f in tqdm.tqdm(all_hparams, desc="Hyp search") :

    train_start = time.time()

    forest = sklearn_RandomForestClassifier(
                    n_estimators=m, # m
                    criterion=f, # f
                    max_depth=r, # r
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features=d_prime, # d'
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.0,
                    bootstrap=True,
                    random_state=0,
                    verbose=0,
                    max_samples=n_prime # n'
                    )


    forest.fit(X_tr, Y_tr)
    train_acc = eval(forest, X_tr, Y_tr)
    val_acc = eval(forest, X_val, Y_val)

    training_time = time.time() - train_start

    # m, r, n_prime, d_prime, f
    key = f"m={m}_r={r}_n'={n_prime}_d'={d_prime}_f={f}"
    print("\n", key, train_acc * 100, val_acc * 100, training_time)

    all_models[key] = forest
    all_performances[key] = [train_acc, val_acc]

with open(f"{log_dir}/all_performances_sktree.pickle","wb") as file_handle:
    pickle.dump(all_performances, file_handle, pickle.HIGHEST_PROTOCOL)
with open(f"{log_dir}/all_models_sktree.pickle","wb") as file_handle:
    pickle.dump(all_models, file_handle, pickle.HIGHEST_PROTOCOL)

x = np.array(list(all_performances.values()))[:,0]
_ = plot_cdf(x, fileName="hp_search_sktree_cdf_train", dpf=DIR_PATH_FIGURES__)
plt.close()
x = np.array(list(all_performances.values()))[:,1]
_ = plot_cdf(x, fileName="hp_search_sktree_cdf_val", dpf=DIR_PATH_FIGURES__)

