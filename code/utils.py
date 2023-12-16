
import numpy as np
import pandas as pd
from collections import Counter
import string
import os
import ntpath
import copy

# Default image size
H, W = 28, 28

# To save data, figures, model checkpoints ...
MAIN_PATH="../"

# Path of train.csv and test.csv
DATA_PATH=f"{MAIN_PATH}/data"
# # Where to save thes figures
DIR_PATH_SUBMISSIONS = f"{MAIN_PATH}/submissions"
# Where to save thes figures
DIR_PATH_FIGURES = f"{MAIN_PATH}/figures"
# # Where to save thes figures
# DIR_PATH_SUBMISSIONS = "./submissions"
# # Where to save thes figures
# DIR_PATH_FIGURES = "./figures"

os.makedirs(DIR_PATH_SUBMISSIONS, exist_ok=True)
os.makedirs(DIR_PATH_FIGURES, exist_ok=True)

"""# Utils SUBMISSIONS"""

#UPPER_CASE_ALPHABETS = list(string.ascii_uppercase.replace("J", "").replace("Z", ""))
UPPER_CASE_ALPHABETS = list(string.ascii_uppercase)
UPPER_CASE_ASCII = [ord(char) for char in UPPER_CASE_ALPHABETS]

def remove_J_Z(Y) :
    # remove 9 (J) and 25 (Z)
    Y = copy.deepcopy(Y)
    Y[Y > 9] = Y[Y > 9] - 1
    return Y

def reconsider_J_Z(Y) :
    # reconsider 9 (J) and 25 (Z)
    Y = copy.deepcopy(Y)
    Y[Y >= 9] = Y[Y >= 9] + 1
    return Y  

def class_to_ascii(Y) :
    # reconsider 9 (J) and 25 (Z) and convert to ascii
    return [UPPER_CASE_ASCII[y] for y in reconsider_J_Z(Y)]

def chars2ascii_sum_ascii2char(character1, character2, is_int_or_char = "int"):
    """
    convert two characters to ascii, sum them, reconvert the result in string format
    Note :
    * The labels on Image A and Image B for every ID are only in upper case alphabets (A-Z).
    * The final resultant character after the ASCII sum and conversion can be both in upper and lower
      case characters (including special characters).
    * If the ASCII sum exceeds 122 (ASCII for 'z') then you are expected to subtract the lower bound ASCII - 65 (ASCII for 'A')
      from the computed sum until your resultant value is within the range of 65-122 ASCII value.
    * The final character computed must be converted to string dtype for consistency.
    """
    assert is_int_or_char in  ["char", "int"]
    if is_int_or_char == "int" :
        assert (character1 in UPPER_CASE_ASCII) and (character2 in UPPER_CASE_ASCII), f"{min(UPPER_CASE_ASCII)} <= {character1}, {character2} <= {max(UPPER_CASE_ASCII)}"
    else :
        assert (character1 in UPPER_CASE_ALPHABETS) and (character2 in UPPER_CASE_ALPHABETS), f"{min(UPPER_CASE_ALPHABETS)} <= {character1}, {character2} <= {max(UPPER_CASE_ALPHABETS)}"
        character1, character2 = ord(character1), ord(character2)
    sum_ascii = character1+character2
    while sum_ascii > 122 : sum_ascii -= 65
    return chr(sum_ascii)

def ascii2char_list(list_):
    """
    convert two list  ascii to int
    """
    return [chr(char) for char in list_]

def chars2ascii_sum_ascii2char_list(list_chars1, list_chars2, is_int_or_char = "int"):
    """
    convert two list characters to ascii, sum them, reconvert the result in string format
    """
    return [chars2ascii_sum_ascii2char(char1, char2, is_int_or_char) for char1, char2 in zip(list_chars1, list_chars2)]

def save_for_submission(IDs, Y_hat_A, Y_hat_B, fileName="submission.csv", dps=None) :
    """Save the prediction in an appropriate format for submission"""
    Y_hat_A = class_to_ascii(Y_hat_A)
    Y_hat_B = class_to_ascii(Y_hat_B)
    Y_hat = chars2ascii_sum_ascii2char_list(Y_hat_A, Y_hat_B, is_int_or_char = "int")
    print(Counter(Y_hat))
    if dps is None : dps = DIR_PATH_SUBMISSIONS
    pd.DataFrame({'id': IDs, 'label': Y_hat}, dtype=str).to_csv(os.path.join(dps, fileName),  index=False, sep=",")

def predict_test(model, X_test) :
    """make prediction on test set using a model"""
    return model.predict(X_test[0]), model.predict(X_test[1])

def predict_nontest(model, X, Y, seed=0) :
    """split (X, Y) in two parts, make prediciton like in test set"""
    n = X.shape[0]
    mid = n//2

    X = copy.deepcopy(X)
    Y = copy.deepcopy(Y)
    if 'torch' in str(type(X))  : 
        import torch
        torch.manual_seed(seed)
        indices = torch.randperm(n)
    elif  'numpy' in str(type(X))  :
        np.random.seed(seed) 
        indices = np.random.permutation(n)
    X, Y = X[indices], Y[indices] 

    Y_hat = model.predict(X)
    test_acc = sum(Y_hat == Y) / n
    if 'torch' in str(type(test_acc)) : test_acc = test_acc.item()

    Y_tmp = class_to_ascii(Y_hat)
    Y_hat_A, Y_hat_B = Y_tmp[:mid], Y_tmp[mid:2*mid+1]
    Y_hat_char = chars2ascii_sum_ascii2char_list(Y_hat_A, Y_hat_B, is_int_or_char = "int")

    Y_tmp = class_to_ascii(Y)
    Y_A, Y_B = np.array(Y_tmp[:mid]),  np.array(Y_tmp[mid:2*mid])
    Y_char = chars2ascii_sum_ascii2char_list(Y_A, Y_B, is_int_or_char = "int")

    test_acc_sum = sum([int(Y_char[i] == Y_hat_char[i]) for i in range(len(Y_char))]) / len(Y_char)

    return test_acc*100, test_acc_sum*100

def eval(model, X, Y):
    """Evaluate the model accuracy on (X, Y)"""
    Y_hat = model.predict(X)
    return sum(Y_hat == Y) / X.shape[0]

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

def path_leaf(path):
    # https://stackoverflow.com/a/8384788/11814682
    head, tail = ntpath.split(path)
    return head