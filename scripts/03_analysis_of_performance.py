from sklearn.metrics import precision_recall_fscore_support as score
import tensorflow as tf
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Use LaTEX font
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Palatino']})


def plot_confusion_matrix_inner(y_true, y_pred, classes,
                                title=None,
                                cmap=plt.cm.binary):

    np.set_printoptions(precision=2)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # Label the ticks with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax


def plot_confusion_matrix(model, X, y, label_tokenizer, title="Confusion Matrix on Test Set"):
    y_pred = np.argmax(model.predict(X), axis=-1)
    y_true = y  # np.argmax(y, axis=1)

    label_map = {v-1: k for k, v in label_tokenizer.word_index.items()}
    class_names = [label_map[i] for i in range(8)]
    class_names = ["Other" if name == "other" else name.upper()
                   for name in class_names]

    plot_confusion_matrix_inner(y_true, y_pred, classes=class_names,
                                title=title, cmap=plt.cm.binary)
    plt.show()


def plot_scores_graph(model, X, y, label_tokenizer):
    test_pred = np.argmax(model.predict(X), axis=1)
    cm = confusion_matrix(y, test_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    precision, recall, _, _ = score(y, test_pred)
    accuracy = cm.diagonal()
    f1 = f1_score(y, test_pred, average=None)
    label_map = {v-1: k for k, v in label_tokenizer.word_index.items()
                 }  # Need the dict
    class_names = [label_map[i] for i in range(8)]
    class_names = ["Other" if name == "other" else name.upper()
                   for name in class_names]
    accuracy = list(map(lambda x: int(round(x, 0)), accuracy*100))
    precision = list(map(lambda x: int(round(x, 0)), precision*100))
    recall = list(map(lambda x: int(round(x, 0)), recall*100))
    f1 = list(map(lambda x: int(round(x, 0)), f1*100))
    x = np.arange(len(class_names))  # the label locations
    width = 0.21
    fig, ax = plt.subplots()
    ax.set_ylim(top=100)
    rects1 = ax.bar(x - 2*width, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x - width, precision, width, label='Precision')
    rects3 = ax.bar(x, recall, width, label='Recall')
    rects4 = ax.bar(x + width, f1, width, label='F1')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score (%)')
    ax.set_title('Classification metric scores by TED label')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    fig.tight_layout()
    plt.show()
