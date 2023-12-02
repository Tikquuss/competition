import numpy as np
import pandas as pd
import cv2
from collections import Counter
import os

from utils import DATA_PATH, H, W
from utils import remove_J_Z

def over_sampling(data) :
    """Over sample the data"""
    df = pd.DataFrame(data=data)
    label_idx = 16
    #n_0, n_1, n_2 = df[label_idx].value_counts()
    n_012 = Counter(df[label_idx])
    class_max = max(n_012 , key=n_012.get)
    n_max = n_012[class_max]

    df_under = df[df[label_idx] == class_max]
    for i in n_012.keys() :
        if i != class_max :
            # https://stackoverflow.com/a/62873637/11814682
            df_class_i_under = df[df[label_idx] == i].sample(n_max, replace=True)
            df_under = pd.concat([df_under,  df_class_i_under], axis=0)

    data = df_under.to_numpy()
    print(Counter(data[:,label_idx]))
    return data

def under_sampling(data) :
    """Undersample sample the data"""
    df = pd.DataFrame(data=data)
    label_idx = 16
    #n_0, n_1, n_2 = df[label_idx].value_counts()
    n_012 = Counter(df[label_idx])
    class_min = min(n_012 , key=n_012.get)
    n_min = n_012[class_min]

    df_over = df[df[label_idx] == class_min]
    for i in n_012.keys() :
        if i != class_min :
            df_class_i_over = df[df[label_idx] == i].sample(n_min, replace=False)
            df_over = pd.concat([df_over,  df_class_i_over], axis=0)

    data = df_over.to_numpy()
    print(Counter(data[:,label_idx]))
    return data

class StandardScaler(object):
    """Transforms the data to have 0 mean and unit variance on each feature
    https://kenzotakahashi.github.io/scikit-learns-useful-tools-from-scratch.html
    """
    def __init__(self):
        pass

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X - self.mean_, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MinMaxScaler(object):
    """
    Scale the data in certain ranges
    https://kenzotakahashi.github.io/scikit-learns-useful-tools-from-scratch.html
    """
    def __init__(self, feature_range=(0, 1)):
        self.low_, self.high_ = feature_range

    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        X_std = (X - self.min_) / (self.max_ - self.min_)
        return X_std * (self.high_ - self.low_) + self.low_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def resize_data(X, HEIGHT = 10, WIDTH = 10):
    X_img = X.reshape(-1, H, W) / 255 # (n, H, W)
    #i = np.random.randint(0, X.shape[0])
    #plt.imshow(X_img[i]), plt.show()
    X_resize = np.stack([cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC) for img in X_img]) # (n, HEIGHT, WIDTH)
    #plt.imshow(X_resize[i]), plt.show()
    return X_resize.reshape(-1, HEIGHT*WIDTH)

def get_train_val_test_set(
    train_data, holdout_test_data, val_data, X_test, d, do_over_sampling=False, do_under_sampling=False,
    HEIGHT=None, WIDTH=None,
    scaler_class = None,
    is_pytorch=False,
    device='cpu'
    ) :
    """Apply or not over/under sampling, separate data in (X, Y) form, apply normalization"""

    assert scaler_class in [None, 'standard_scaler', 'min_max_scaler']
    if scaler_class == 'standard_scaler' : scaler_class_inst = StandardScaler
    elif scaler_class == 'min_max_scaler' : scaler_class_inst = MinMaxScaler

    ##############
    if do_over_sampling : train_data = over_sampling(train_data)
    if do_under_sampling : train_data = under_sampling(train_data)
    #############

    # train and val data
    X_train, Y_train = train_data[:,1:], train_data[:,0].astype(int)
    X_holdout_test, Y_holdout_test = holdout_test_data[:,1:], holdout_test_data[:,0].astype(int)
    X_val, Y_val = val_data[:,1:], val_data[:,0].astype(int)
    
    # X_test
    X_A, X_B = X_test[:,:d], X_test[:,d:] # (n, d)

    ### all data
    #data_all = dataset[indices]
    data_all = np.concatenate((train_data, val_data), axis=0)
    X_all, Y_all = data_all[:,1:], data_all[:,0].astype(int)
    X_holdout_test_all = X_holdout_test
    X_A_all, X_B_all = X_A, X_B

    # remove 9 (J) and 25 (Z)
    Y_train = remove_J_Z(Y_train)
    Y_holdout_test = remove_J_Z(Y_holdout_test)
    Y_val = remove_J_Z(Y_val)
    Y_all = remove_J_Z(Y_all)
    
    ############## resize
    if HEIGHT is not None and  WIDTH is not None :
        X_train = resize_data(X_train, HEIGHT, WIDTH)
        X_holdout_test = resize_data(X_holdout_test, HEIGHT, WIDTH)
        X_val = resize_data(X_val, HEIGHT, WIDTH)
        X_A = resize_data(X_A, HEIGHT, WIDTH)
        X_B = resize_data(X_B, HEIGHT, WIDTH)

        X_all = resize_data(X_all, HEIGHT, WIDTH)
        X_holdout_test_all = resize_data(X_holdout_test_all, HEIGHT, WIDTH)
        X_A_all = resize_data(X_A_all, HEIGHT, WIDTH)
        X_B_all = resize_data(X_B_all, HEIGHT, WIDTH)
    ##############

    ############## preprocessing the data
    if scaler_class is not None :
        scaler = scaler_class_inst().fit(X_train)
        if scaler_class == 'standard_scaler' : print("normal, mean, std :", scaler.mean_[:5], scaler.scale_[:5])
        elif scaler_class == 'min_max_scaler' : print("normal, min, max :", scaler.min_[:5], scaler.max_[:5])
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_holdout_test = scaler.transform(X_holdout_test)
        X_A = scaler.transform(X_A)
        X_B = scaler.transform(X_B)

        scaler = scaler_class_inst().fit(X_all)
        if scaler_class == 'standard_scaler' :print("all, mean, std :", scaler.mean_[:5], scaler.scale_[:5])
        elif scaler_class == 'min_max_scaler' : print("all, min, max :", scaler.min_[:5], scaler.max_[:5])
        X_all = scaler.transform(X_all)
        X_A_all = scaler.transform(X_A_all)
        X_B_all = scaler.transform(X_B_all)
        X_holdout_test_all = scaler.transform(X_holdout_test_all)

    # For pytorch
    if is_pytorch :
        import torch
        DTYPE=torch.float64
        if HEIGHT is None : HEIGHT = H
        if WIDTH is None : WIDTH = W
        X_train, Y_train = torch.from_numpy(X_train).to(DTYPE).reshape(-1, 1, HEIGHT, WIDTH).to(device), torch.from_numpy(Y_train).long().to(device)
        X_holdout_test, Y_holdout_test = torch.from_numpy(X_holdout_test).to(DTYPE).reshape(-1, 1, HEIGHT, WIDTH).to(device), torch.from_numpy(Y_holdout_test).long().to(device)
        X_val, Y_val = torch.from_numpy(X_val).to(DTYPE).reshape(-1, 1, HEIGHT, WIDTH).to(device), torch.from_numpy(Y_val).long().to(device)
        X_all, Y_all = torch.from_numpy(X_all).to(DTYPE).reshape(-1, 1, HEIGHT, WIDTH).to(device), torch.from_numpy(Y_all).long().to(device)
        X_A, X_B = torch.from_numpy(X_A).to(DTYPE).reshape(-1, 1, HEIGHT, WIDTH).to(device), torch.from_numpy(X_B).to(DTYPE).reshape(-1, 1, HEIGHT, WIDTH).to(device)
        X_A_all, X_B_all = torch.from_numpy(X_A_all).to(DTYPE).reshape(-1, 1, HEIGHT, WIDTH).to(device), torch.from_numpy(X_B_all).to(DTYPE).reshape(-1, 1, HEIGHT, WIDTH).to(device)
        X_holdout_test_all = torch.from_numpy(X_holdout_test_all).to(DTYPE).reshape(-1, 1, HEIGHT, WIDTH).to(device)

    X_test = X_A, X_B
    X_test_all = X_A_all, X_B_all
    return X_train, Y_train, X_holdout_test, Y_holdout_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_holdout_test_all, d

def train_test_split(
    train_dataset, test_dataset, train_pct, holdout_pct, 
    do_over_sampling=False, do_under_sampling=False,
    HEIGHT=None, WIDTH=None,
    scaler_class=None,
    is_pytorch=False,
    device='cpu',
    seed=0) :
    """Standard train/val/test split"""

    np.random.seed(seed)

    dataset = train_dataset.to_numpy()
    d = dataset.shape[-1]-1 #
    print(d)

    # https://stackoverflow.com/a/3677283/11814682
    n = dataset.shape[0]
    train_size = int(train_pct * n / 100)
    holdout_test_size = int(holdout_pct * n / 100)
    val_size = n - train_size - holdout_test_size
    indices = np.random.permutation(n)

    train_idx = indices[:train_size]
    #holdout_test_idx, val_idx = indices[train_size:train_size+holdout_test_size], indices[train_size+holdout_test_size:]
    dummy_idx = indices[:2]
    if holdout_test_size==0 :
        if val_size!=0: holdout_test_idx, val_idx = dummy_idx, indices[train_size:]
        else : holdout_test_idx, val_idx = dummy_idx, dummy_idx
    else :
        if val_size!=0: 
            holdout_test_idx, val_idx = indices[train_size:train_size+holdout_test_size], indices[train_size+holdout_test_size:]
        else : holdout_test_idx, val_idx = indices[train_size:], dummy_idx
        
    train_data, holdout_test_data, val_data = dataset[train_idx,:], dataset[holdout_test_idx,:], dataset[val_idx,:]

    return get_train_val_test_set(
        train_data, holdout_test_data, val_data, test_dataset.to_numpy(), d, do_over_sampling, 
        do_under_sampling, HEIGHT, WIDTH, scaler_class=scaler_class, is_pytorch=is_pytorch, device=device)

def train_test_split_k_fold(
    train_dataset, test_dataset, train_pct, holdout_pct, 
    do_over_sampling=False, do_under_sampling=False,
    HEIGHT=None, WIDTH=None,
    scaler_class=None,
    is_pytorch=False,
    device='cpu',
    seed=0) :
    """Split the (train+validation) data in multiple folds for k-fold cross validation"""

    np.random.seed(seed)

    #df = train_dataset.copy()
    df = train_dataset#.drop(COLUMS_TO_REMOVE, axis=1)
    d = df.shape[-1]-1
    print(d)
    # shuffle
    df = df.reindex(np.random.permutation(df.index))
    # reset index
    df = df.reset_index(drop=True)
    n = len(df)
    train_size = train_pct * n // 100
    holdout_test_size = holdout_pct * n // 100
    val_size = n - train_size - holdout_test_size

    if holdout_test_size==0 :
        df_holdout = df[:1]
    else :
        df_holdout = df[:holdout_test_size]
        df = df[holdout_test_size:]
    n = n - holdout_test_size

    n_folds = n // val_size + 1 * (0 if n % val_size == 0 else 1)
    fold = pd.DataFrame()
    for i in range(n_folds) :
        start_val = i*val_size
        start_train = (i+1)*val_size
        val_fold = df.loc[start_val:start_val+val_size-1]
        train_fold = pd.concat([fold, df.loc[start_train:] ])
        fold = pd.concat([fold, val_fold])
        yield get_train_val_test_set(train_fold.to_numpy(), df_holdout.to_numpy(), val_fold.to_numpy(), test_dataset.to_numpy(), d,
                                     do_over_sampling, do_under_sampling, HEIGHT, WIDTH, scaler_class=scaler_class, is_pytorch=is_pytorch, device=device)


def get_dataset(
    train_pct=80, holdout_pct=10, k_fold=False, HEIGHT=None, WIDTH=None, 
    do_over_sampling=False, do_under_sampling=False,
    scaler_class=None,
    is_pytorch=False,
    device='cpu',
    seed=0,
    just_dataframe=False
    ) :

    COLUMS_TO_REMOVE = ["id"]
    # label, pixel1, ..., pixel784 (single 28x28=784 pixel image with grayscale values between 0-255)
    train_dataset = pd.read_csv(os.path.join(DATA_PATH, "sign_mnist_train.csv"))
    # ALL_COLUMS = test_dataset.columns
    # id,	pixel_a1, ...,	pixel_a784,	pixel_b1,	..., pixel_b784
    test_dataset = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    IDs_test = test_dataset["id"].to_numpy()
    test_dataset = test_dataset.drop(COLUMS_TO_REMOVE, axis=1)

    if just_dataframe : return train_dataset, test_dataset, IDs_test

    if not k_fold :
        X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d = train_test_split(
            train_dataset, test_dataset, train_pct=train_pct, holdout_pct=holdout_pct, 
            do_over_sampling=do_over_sampling, do_under_sampling=do_under_sampling, HEIGHT=HEIGHT, WIDTH=WIDTH,
            scaler_class=scaler_class, is_pytorch=is_pytorch, device=device, seed=seed
            )

        print("print(len(train_dataset), X_tr.shape, X_ht_test.shape, X_val.shape, X_test[0].shape, X_test[1].shape)")
        print(len(train_dataset), X_tr.shape, X_ht_test.shape, X_val.shape, X_test[0].shape, X_test[1].shape)
        #print(X_all.shape, X_test_all[0].shape)

        return IDs_test, (X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d)

    else :
        kfold_iterator = list(train_test_split_k_fold(
            train_dataset, test_dataset, train_pct=train_pct, holdout_pct=holdout_pct, 
            do_over_sampling=do_over_sampling, do_under_sampling=do_under_sampling, HEIGHT=HEIGHT, WIDTH=WIDTH,
            scaler_class=scaler_class, is_pytorch=is_pytorch, device=device, seed=seed))

        print("n folds : ", len(kfold_iterator))
        for X_tr_kf, Y_tr_kf, X_ht_test_kf, Y_ht_test_kf, X_val_kf, Y_val_kf, X_all_kf, Y_all_kf, X_test_kf, X_test_all_kf, X_ht_test_all_kf, d in kfold_iterator :
            print("X_tr_kf.shape, X_ht_test_kf.shape, X_val_kf.shape, X_all_kf.shape")
            print(X_tr_kf.shape, X_ht_test_kf.shape, X_val_kf.shape, X_all_kf.shape)
            break

        return IDs_test, kfold_iterator

if __name__ == "__main__" :
    import os
    import matplotlib.pyplot as plt


    H, W = 28, 28
    COLUMS_TO_REMOVE = ["id"]

    # label, pixel1, ..., pixel784 (single 28x28=784 pixel image with grayscale values between 0-255)
    train_dataset = pd.read_csv(os.path.join(DATA_PATH, "sign_mnist_train.csv"))
    # ALL_COLUMS = test_dataset.columns
    # id,	pixel_a1, ...,	pixel_a784,	pixel_b1,	..., pixel_b784
    test_dataset = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    IDs_test = test_dataset["id"].to_numpy()
    test_dataset = test_dataset.drop(COLUMS_TO_REMOVE, axis=1)

    # (785-1)*2+1 = 1569
    print(train_dataset.shape, test_dataset.shape)


    print(Counter(pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))['label']))

    """**Statistics**"""

    ct = Counter(train_dataset['label'])
    print(min(ct.keys()), max(ct.keys()))
    ct = dict(sorted(ct.items()))
    s = sum(ct.values())
    pr = {k : round(100*ct[k]/s, 3) for k in ct.keys()}
    print(ct)
    print(pr)
    sum(pr.values())

    rows, cols = 1, 2
    figsize = (6, 4)
    figsize=(cols*figsize[0], rows*figsize[1])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(rows, cols, 1)

    ax.hist(
        x=train_dataset['label'].to_numpy(),
        label=None, color=None,
        #histtype='bar',
        #histtype='barstacked',
        #histtype='step',
        histtype='stepfilled',
        align='mid', orientation='vertical',
        stacked=False,
        density=True,
        rwidth=None,
        #rwidth=0.8,
    )

    plt.show()

    """**Visualize an example**"""

    i = np.random.randint(0, train_dataset.shape[0])
    img = train_dataset.iloc[i][1:].to_numpy().reshape(H, W) / 255
    plt.imshow(img)

    """## data"""

    HEIGHT, WIDTH = 10, 10
    HEIGHT, WIDTH = 5, 5
    #HEIGHT, WIDTH = 2, 2

    #HEIGHT, WIDTH = None, None

    """**Normal training data**"""

    X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d = train_test_split(train_dataset, test_dataset, train_pct=70, holdout_pct=10, do_over_sampling=False, do_under_sampling=False, HEIGHT=HEIGHT, WIDTH=WIDTH)
    #X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d = train_test_split(train_dataset, test_dataset, train_pct=70, holdout_pct=10, do_over_sampling=False, do_under_sampling=False, HEIGHT=HEIGHT, WIDTH=WIDTH)

    #X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d = train_test_split(train_dataset, test_dataset, train_pct=100, holdout_pct=0, do_over_sampling=False, do_under_sampling=False, HEIGHT=HEIGHT, WIDTH=WIDTH)

    print(len(train_dataset), X_tr.shape, X_ht_test.shape, X_val.shape, X_test[0].shape, X_test[1].shape)
    print(X_all.shape, X_test_all[0].shape)

    """**Training data for k-fold cross validation**"""