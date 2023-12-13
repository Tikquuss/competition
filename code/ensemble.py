import numpy as np
from plotter import confusion_matrix, scores, plot_confusion_matrix, show_example_images

def ensemble(all_Y, Y_start=None):
    """Select the most frequent prediction as the best prediction"""
    all_Y = [Y[:,np.newaxis] for Y in all_Y]
    Y = np.concatenate(all_Y, axis=1) # (n_test, n_fold)
    Y = np.array([np.argmax(np.bincount(Y[i])) for i in range(Y.shape[0])]) # (n_test)
    if Y_start is None : return Y
    conf_matrix = confusion_matrix(Y_start, Y, n_classes=c)
    plot_confusion_matrix(conf_matrix)
    print(scores(conf_matrix))
    return Y


