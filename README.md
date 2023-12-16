# IFT6390B: Fundamentals of machine learning

<hr>
<center><b>
Kaggle Competition 2 : ASCII Sign Language
</b></center><hr>
<br>

* Name : TIKENG NOTSAWO Pascal Junior 
* Email : pascal.junior.tikeng.notsawo@umontreal.ca
* Matricule (student number) : 20224267 
* Kaggle username: pascalnotsawo

* Kaggle : https://www.kaggle.com/competitions/ascii-sign-language/overview

<br>

## Overwiew

Our code is divided into files. Here is the role of each file:
* General :
    * `data.py`: contains useful routines concerning data (over/under sampling methods, standardization method ...)
    * `ensemble.py`: method for combining the decisions of several classifiers
    * `plotter.py`: functions for displaying images, figures, ...
    * `utils.py`: contains useful routines for all the code

* Random Rorest : 
    - `model_tree.py`: contains our numpy implementation of random forest
    - `train_tree.py`: the script to be used to train a random forest
    - `hp_search_sklearn.py`: the script to run the hyperparameter search for decision trees (with sklearn implementation)

* Convolutional Neural Network
    - `model_cnn.py`: Implementation of our CNN models (smallCNN, bigCNN, RestNet ...)
    - `ResNet_general.py`: (we didn't use this): an implementation of ResNet50, ResNet101 and ResNet152 from https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
    - `train_cnn.py` : script to be used to train CNNs
    - `train_loop.sh`: train with several random seeds
    - `trainer.py` : contains the main Trainer class, which takes a model, its loss and its optimizer, trains it by keeping the parameters at each epoch, and returns the model

## Requirements 

```txt
torch
torchvision
matplotlib
numpy
pandas
scikit-learn
opencv-python
tqdm
```

```bash
git clone https://github.com/Tikquuss/competition
cd competition/code
pip install -r ../requirements.txt
```

## Execution

* To train random forest on the command line, call:

```bash
python train_tree.py --model_name test --n_estimators 100 --max_depth 100 --max_samples 1.0 --max_features sqrt --sklearn False --SIZE 28 --train_pct 90 --holdout_pct 10 --seed 0

% model_name : to save the submission
% n_estimators : number of trees
% max_depth : maximum deph of trees
% criterion : criterion (gini, entropy)
% max_samples : max boostrap samples
% max_features : max features per tree 
% sklearn : Try sklearn or not
% train_pct : training data percentage
% holdout_pct : test data percentage
% SIZE : HEIGHT, WIDTH (upsample?, downsample?)
% seed : random seed
```

* To train CNNs on the command line, call:

```bash
python train_cnn.py --model_name test  --learning_rate 0.001 --weight_decay 0.0001 --scheduler False --n_epochs 50 --batch_size 512 --dropout_conv 0.0 --dropout_fc 0.0 --train_pct 90 --holdout_pct 10 --seed 0

% model_name : to save the submission
% learning_rate : learning rate
% weight_decay : weight decay
% scheduler :  use exponential lr scheduler or not
% n_epochs : number of epochs
% batch_size : bach size
% dropout_conv : dropout for convnet 
% dropout_fc : dropout classifier (fc layer)
% train_pct : training data percentage
% holdout_pct : test data percentage
% SIZE : HEIGHT, WIDTH (upsample?, downsample?)
% seed : random seed
```
OR (to loop over many random seeds)
```bash
chmod +x train_loop.sh
./train_loop.sh test 2
% the first argument is the model name and the second is the number of epochs
% see the file for more informations
```

## Performances & Submission & Figures

Once training is complete, will : :
* display training performance, and :
    - validation performance if there was validation data (if train_pct + holdout_pct < 100)
    - test performance if there was test data (if holdout_pct > 0)
* display training time
* save (in the folder containing the code) :
    - training curve (for CNNs)
    - confusion matrices
    - csv file for submission to kaggle
    - the trained model and its information

## Notebooks

In the folder [notebooks](notebooks), notebook [data_informations.ipynb](notebooks/data_informations.ipynb) allows to reproduce some figures concerning data information, and notebooks [RandomForest&Sklearn.ipynb](notebooks/RandomForest&Sklearn.ipynb) and [CNN.ipynb](notebooks/CNN.ipynb) show respectively how to interactively execute random forest and CNN model training, and also reproduce notebook figures