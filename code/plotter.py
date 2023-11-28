import matplotlib.pyplot as plt
import numpy as np

from utils import DIR_PATH_FIGURES

def plot_training_curve(n_epochs, train_losses, train_accs, val_losses, val_accs, fileName = None):
    """Plot the training/validation losses & acc per epoch of training"""
    rows, cols = 1, 2
    figsize = (6, 4)
    figsize=(cols*figsize[0], rows*figsize[1])
    fig = plt.figure(figsize=figsize)
    type_c = '-o'
    type_c = '-'

    epochs = np.arange(n_epochs) + 1
    
    ax = fig.add_subplot(rows, cols, 1)
    linewidth = 0.2
    ax.plot(epochs, train_losses, type_c, label="train")
    if val_losses : ax.plot(epochs, val_losses, type_c, label="val")
    ax.tick_params(axis='y')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()
    ax.grid()

    ax = fig.add_subplot(rows, cols, 2)
    linewidth = 0.2
    ax.plot(epochs, train_accs, type_c, label="train")
    if val_accs : ax.plot(epochs, val_accs, type_c, label="val")
    ax.tick_params(axis='y')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.legend()
    ax.grid()

    if fileName is not None : plt.savefig(f"{DIR_PATH_FIGURES}/{fileName}"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.show()


def plot_cdf(samples, label = None, ax = None, fileName=None):
    """Plot the cumulative distributuion"""
    samples = np.sort(samples)
    y = np.arange(len(samples))/float(len(samples))
    if ax :
        ax.plot(samples, y, label=label)
        ax.grid()
    else :
        plt.plot(samples, y, label=label)
        plt.grid()
    if fileName is not None : plt.savefig(f"{DIR_PATH_FIGURES}/{fileName}"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    return samples, y

def custom_imshow(img_data, ax=None, fig=None, add_text=False, n_decimals=2,
                  xticklabels=None, yticklabels=None, x_label=None, y_label=None,
                  rotation_x=0, rotation_y=0,
                  imshow_kwarg = {},
                  colorbar = True,
                  show=True,
                  fileName=None,) :

    """"
    Custom plt.imshow
    This helps to plot the heatmap of performances as a function of many hyperparameters
    """
    if (ax is None) and (fig is None) :
        L, C = 1, 1
        figsize=(C*32, L*4)
        #figsize=(C*6, L*4)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(L, C, 1)

    img=ax.imshow(img_data, **imshow_kwarg)
    if add_text :
        for (j, i), label in np.ndenumerate(img_data):
            ax.text(i, j, round(label, n_decimals), ha='center', va='center', fontsize=20)
    if colorbar : fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, aspect=20)

    if xticklabels is not None :
        ax.set_xticks(list(range(len(xticklabels))) )
        ax.set_xticklabels(xticklabels, rotation=rotation_x, fontsize=20)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=False, labeltop=True)
    if x_label : ax.set_xlabel(x_label, fontsize=20)

    if yticklabels is not None :
        ax.set_yticks(list(range(len(yticklabels))) )
        ax.set_yticklabels(yticklabels, rotation=rotation_y, fontsize=20)
        ax.tick_params(axis="y", bottom=True, top=False, labelbottom=True, labeltop=False)
    if y_label : ax.set_ylabel(y_label, fontsize=20)

    if fileName is not None : plt.savefig(f"{DIR_PATH_FIGURES}/{fileName}"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    if show : plt.show()

    return img

def confusion_matrix(true_labels, pred_labels, n_classes = None):
    """Compute the confusion matrix"""
    if n_classes is None : n_classes = int(max(true_labels))
    matrix = np.zeros((n_classes, n_classes))
    for (true, pred) in zip(true_labels, pred_labels): matrix[int(true), int(pred)] += 1
    return matrix

def scores(cm, fileName=None) :
    """Compute and plot the per class (and the average) precision, recall and  f1-score using the confusion matrix cm"""
    precision_c = np.diag(cm) / np.sum(cm, axis = 0)
    recall_c = np.diag(cm) / np.sum(cm, axis = 1)
    f1_score_c = 2 * precision_c * recall_c / (precision_c+recall_c)
    precision, recall, f1_score = np.mean(precision_c), np.mean(recall_c), np.mean(f1_score_c)

    img_data = np.array([
            precision_c.tolist() + [precision],
            recall_c.tolist() + [recall],
            f1_score_c.tolist() + [f1_score]
        ])

    img_data = img_data*100
    custom_imshow(img_data, add_text=True, n_decimals=2,
                  xticklabels=[f'{i}' for i in range(len(cm))]  + ['mean'],
                  yticklabels=['precision', 'recall', 'f1_score'],
                  x_label=None, y_label=None,
                  rotation_x=0, rotation_y=0,
                  fileName=fileName)

    return {"acc" : np.diag(cm).sum() / cm.sum(), "precision" : precision_c, "recall" : recall_c, "f1_score" : f1_score_c }
    #return None

def plot_confusion_matrix(conf_matrix, fileName=None) :
      """Plot the confusion matrix"""
      L, C = 1, 1
      figsize=(C*15, L*10)
      #figsize=(C*6, L*4)
      fig = plt.figure(figsize=figsize)
      ax = fig.add_subplot(L, C, 1)

      conf_matrix  = conf_matrix.astype(int)
      ax.imshow(conf_matrix)
      for (j,i),label in np.ndenumerate(conf_matrix):
          ax.text(i, j, label, ha='center', va='center')

      tmp = [i for i in range(len(conf_matrix))]
      tmp_t = list(range(len(tmp)))

      ax.set_xticks(tmp_t)
      ax.set_xticklabels(tmp, rotation=90)
      ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

      ax.set_yticks(tmp_t)
      ax.set_yticklabels(tmp, rotation=90)
      ax.tick_params(axis="y", bottom=True, top=False, labelbottom=True, labeltop=False)

      if fileName is not None : plt.savefig(f"{DIR_PATH_FIGURES}/{fileName}"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

      plt.show()