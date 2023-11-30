import torch

import tqdm
import copy

from data import get_dataset

class Trainer:
    """Trainer for classification"""
    def __init__(self, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.all_model = {}

    def __call__(self, x):
        return self.model(x)

    def train_loop(self, X, Y, batch_size):
        '''
        Implementation of one training loop accross the data

        :param X : data (n, p)
        :param Y: true labels (one-hot format) (n,)
        :param batch_size : batch size
        '''
        self.model.train()
        n = X.shape[0]
        num_batches = n // batch_size + 1 * (0 if n % batch_size == 0 else 1)
        n_correct = 0
        train_loss = 0.
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X[start_idx : end_idx]
            Y_batch = Y[start_idx : end_idx]

            self.optimizer.zero_grad()
            Y_hat_batch = self.model(X_batch)
            loss = self.criterion(Y_hat_batch, Y_batch)
            loss.backward()
            self.optimizer.step()

            train_loss += loss * Y_hat_batch.shape[0]
            n_correct += torch.sum(Y_hat_batch.argmax(dim=-1) == Y_batch)

        train_loss = train_loss/n
        train_acc = n_correct/n

        return train_acc.item(), train_loss.item()

    def valid_loop(self, X, Y, batch_size):
        '''
        Implementation of one validation loop accross the validation data

        :param X : data (n, p)
        :param Y: true labels (one-hot format) (n,)
        :param batch_size : batch size
        '''
        self.model.eval()
        with torch.no_grad():
            n = X.shape[0]
            num_batches = n // batch_size + 1 * (0 if n % batch_size == 0 else 1)
            n_correct = 0
            val_loss = 0.
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X[start_idx : end_idx]
                Y_batch = Y[start_idx : end_idx]
                Y_hat_batch = self.model(X_batch)
                loss = self.criterion(Y_hat_batch, Y_batch)
                val_loss += loss * Y_hat_batch.shape[0]
                n_correct += torch.sum(Y_hat_batch.argmax(dim=-1) == Y_batch)

            val_loss = val_loss/n
            val_acc = n_correct/n
            return val_acc.item(), val_loss.item()

    def train(self, X_train, Y_train, X_val=None, Y_val=None, batch_size=None, n_epochs = 1, use_tqdm=True):
        '''
        Implementation of the training procedure for n_epochs epochs

        :param X_train : training data (n, p)
        :param Y_train: training labels (n,)
        :param X_val : validation data (n_val, p)
        :param Y_val: validation labels (n_val,)
        :param batch_size : batch size
        '''

        n_train = X_train.shape[0]
        train_batch_size = n_train if batch_size is None else batch_size
        self.test_batch_size = train_batch_size
        do_validation = X_val is not None
        if do_validation :
            n_val = X_val.shape[0]
            val_batch_size = n_val if batch_size is None else batch_size
            self.test_batch_size = val_batch_size

        train_accs, train_losses, val_accs, val_losses = [], [], [], []

        best_model = None
        best_acc = 0.

        self.all_model[-1] = copy.deepcopy(self.model).to('cpu')
        try:
            tmp = range(n_epochs)
            if use_tqdm : tmp = tqdm.tqdm(tmp, desc="Training ...")
            for epoch in tmp:
                try:
                    # Training
                    train_acc, train_loss = self.train_loop(X_train, Y_train, train_batch_size)
                    train_accs.append(train_acc)
                    train_losses.append(train_loss)
                    self.all_model[epoch] = copy.deepcopy(self.model).to('cpu')

                    to_print = f"\n Epoch: {epoch} | Train Acc: {train_acc:.6f} | Train Loss: {train_loss:.6f}"
                    # Validation
                    if not do_validation :
                        if train_acc > best_acc :
                            best_acc = train_acc
                            best_model = copy.deepcopy(self.model)

                    if do_validation :
                        val_acc, val_loss = self.valid_loop(X_val, Y_val, val_batch_size)
                        val_accs.append(val_acc)
                        val_losses.append(val_loss)
                        if val_acc > best_acc :
                            best_acc = val_acc
                            best_model = copy.deepcopy(self.model)

                        to_print += f" | Val Acc: {val_acc:.6f}   | Val Loss: {val_loss:.6f}"
                    if use_tqdm : print(to_print)

                except KeyboardInterrupt:
                    break

        except KeyboardInterrupt:
            pass

        self.model = copy.deepcopy(best_model)

        return train_accs, train_losses, val_accs, val_losses, best_acc

    def test(self, X, batch_size=None):
        '''
        Test the model

        :param X : data (n, p)
        :return the predicted class
        '''
        self.model.eval()
        # Y_hat = self.model(X)
        # return Y_hat.argmax(axis=-1)#.detach().cpu()#.numpy()
        
        with torch.no_grad():
            n = X.shape[0]
            if batch_size is None : 
                try : batch_size = self.test_batch_size
                except AttributeError : batch_size = n
            num_batches = n // batch_size + 1 * (0 if n % batch_size == 0 else 1)
            Y_hat = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X[start_idx : end_idx]
                Y_hat_batch = self.model(X_batch).argmax(axis=-1)
                Y_hat.append(Y_hat_batch)

            try : return torch.concatenate(Y_hat)#.detach().cpu()#.numpy()
            except AttributeError: return torch.cat(Y_hat)#.detach().cpu()#.numpy()

    def predict(self, x):
        return self.test(x)

def get_normal_dataset(
    train_pct=80, holdout_pct=10, 
    train_transforms=None, test_transforms=None, 
    HEIGHT=None, WIDTH=None,
    scaler_class=None,
    device='cpu', seed=0
    ):
    IDs_test, (X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d) = get_dataset(
        train_pct=train_pct, holdout_pct=holdout_pct, k_fold=False, HEIGHT=HEIGHT, WIDTH=WIDTH, do_over_sampling=False, do_under_sampling=False,
        scaler_class=scaler_class, is_pytorch=True, device=device, seed=seed
    )

    if train_transforms is not None :
        X_tr = train_transforms(X_tr)
        X_all = train_transforms(X_all)

    if test_transforms is not None :
        X_ht_test = test_transforms(X_ht_test)
        X_val = test_transforms(X_val)

        X_test = (test_transforms(X_test[0]), test_transforms(X_test[1]))
        X_test_all = (test_transforms(X_test_all[0]), test_transforms(X_test_all[1]))
        X_ht_test_all = test_transforms(X_ht_test_all)

    return IDs_test, (X_tr, Y_tr, X_ht_test, Y_ht_test, X_val, Y_val, X_all, Y_all, X_test, X_test_all, X_ht_test_all, d)


def get_kfold_dataset(
    train_pct=80, holdout_pct=10, 
    train_transforms=None, test_transforms=None, 
    HEIGHT=None, WIDTH=None,
    scaler_class=None,
    device='cpu', seed=0
    ):
    

    IDs_test, kfold_iterator = get_dataset(
        train_pct=train_pct, holdout_pct=holdout_pct, k_fold=True, HEIGHT=HEIGHT, WIDTH=WIDTH, do_over_sampling=False, do_under_sampling=False,
        scaler_class=scaler_class, is_pytorch=True, device=device, seed=seed)

    if (train_transforms is not None) and (test_transforms is not None) :
        return IDs_test, kfold_iterator

    kfold_iterator_tmp = []
    for X_tr_kf, Y_tr_kf, X_ht_test_kf, Y_ht_test_kf, X_val_kf, Y_val_kf, X_all_kf, Y_all_kf, X_test_kf, X_test_all_kf, X_ht_test_all_kf, d in kfold_iterator :

        if train_transforms is not None :
            X_tr_kf = train_transforms(X_tr_kf) #
            X_all_kf = train_transforms(X_all_kf) #

        if test_transforms is not None :
            X_ht_test_kf = test_transforms(X_ht_test_kf) #
            X_val_kf = test_transforms(X_val_kf) #

            X_test_kf = (test_transforms(X_test_kf[0]), test_transforms(X_test_kf[1]))
            X_test_all_kf = (test_transforms(X_test_all_kf[0]), test_transforms(X_test_all_kf[1]))
            X_ht_test_all_kf = test_transforms(X_ht_test_all_kf)

        kfold_iterator_tmp.append(
            X_tr_kf, Y_tr_kf, X_ht_test_kf, Y_ht_test_kf, X_val_kf, Y_val_kf, X_all_kf, Y_all_kf, X_test_kf, X_test_all_kf, X_ht_test_all_kf, d
        )
    return IDs_test, kfold_iterator_tmp

def get_dataset_pytorch(
    train_pct=80, holdout_pct=10, 
    train_transforms=None, test_transforms=None,

    k_fold=False, HEIGHT=None, WIDTH=None, 
    do_over_sampling=False, do_under_sampling=False,
    scaler_class=None,
    is_pytorch=False,
    device='cpu',
    seed=0
    ) :

    torch.manual_seed(seed)

    if not k_fold :
        return get_normal_dataset(
            train_pct=train_pct, holdout_pct=holdout_pct, 
            train_transforms=train_transforms, test_transforms=test_transforms, 
            HEIGHT=HEIGHT, WIDTH=WIDTH,
            scaler_class=scaler_class,
            device=device, seed=seed
            )
    else :
        return get_kfold_dataset(
            train_pct=train_pct, holdout_pct=holdout_pct, 
            train_transforms=train_transforms, test_transforms=test_transforms, 
            HEIGHT=HEIGHT, WIDTH=WIDTH,
            scaler_class=scaler_class,
            device=device, seed=seed
            )