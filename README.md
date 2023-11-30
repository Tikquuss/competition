# IFT6390B: Fundamentals of machine learning

<hr>
<center><b>
Kaggle Competition 2 : ASCII Sign Language
</b></center><hr>
<br><br>

* Name : TIKENG NOTSAWO Pascal Junior 
* Email : pascal.junior.tikeng.notsawo@umontreal.ca
* Matricule (student number) : 20224267 
* Kaggle username: pascalnotsawo

* Kaggle : https://www.kaggle.com/competitions/ascii-sign-language/overview

**Note**: Our codes are well-detailed notebooks with many comments, so there is no need for a very long readme.

## Notebooks

The code is in two notebooks:
- (1) `numpy_LogReg_LinearSVM-l1&l2_Sklearn_Stacknet.ipynb` : it contains the implementation of logistic regression, SVM (linear), MLP (with cross-entropy loss, hinge loss, both vanilla and quadratic version) from scratch, with *numpy*. At the end of the notebook  we also have some sklearn and stacknet methods.
- (2) `pytorch_MLPLogReg_MLPSVM-l1-l2.ipynb` : it contains the MLP implementation in *pytorch* (with cross-entropy loss, hinge loss, both vanilla and quadratic version).

## Requirements 

For notebook 1, we use only simple libraries in the first part (LogReg, SVM, MLP): *numpy* mainly, *scipy* for some calculations (hessian and its inverse, etc.), *pandas* for data, *matplotlib* and *seaborn* for figures, etc.

In the **Sklearn & stacknet** section of the notebook, we use the *sklearn*, *xgboost* and [*pystacknet*](https://github.com/h2oai/pystacknet) libraries.

```bash
# xgboost
pip install xgboost
# pystacknet
!git clone https://github.com/h2oai/pystacknet
%cd pystacknet
!python setup.py install
```

For notebook 2, the only library added is [*Pytorch*](https://pytorch.org/). 

## Execution

Notebooks are generally divided into several sections: 
* global variables: where we define the path to data and the path to the folder where we store figures and checkpoints
* imports: for imports
* data processing: We load data from csvs, perform pre-processing (normalization, over or undersampling, feature selection) and separate into pairs $(X, Y)$: training data, validation data, holdout test data, and test data.
* For each model: detailed implementation and how to train them.

For the model execution section, there's a section for hyperparameters search and another for kfold cross-validation.

Everything is really detailed in the notebooks in each section, so to explain it all again here would be very redundant: all you have to do is make sure you have the necessary libraries, define your global variables as required, and execute the notebooks section by section (or directly the section you're interested in) following the instructions.

Our models are very simple, so it doesn't usually take more than 1 minute to train a model: the hyperparameter search takes time (on the order of minutes) depending on how you define your ranges of values.

## Figures

Functions for displaying and/or saving figures are available in notebooks:
- confusion matrix
- precision, recall and f1-score per class
- training curve per epoch
- model weights (heatmap)
- heatmap of accuracy as a function of several hyperparameters simultaneously
- etc

Figures are saved in a folder of your choice.

For histogram, see the file [plot.py](plot.py)

## Predictions

We have a function that takes a model, evaluates it, and saves the prediction in a Kaggle-submittable format.
