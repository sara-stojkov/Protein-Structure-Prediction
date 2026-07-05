import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


def q3_accuracy(preds, labels, mask):
    """preds, labels, mask: np.array shape (n, seq_len)"""
    correct = (preds == labels) & (mask == 1)
    total = (mask == 1).sum()
    return correct.sum() / total


def f1_per_class(preds, labels, mask):
    p = preds[mask == 1]
    l = labels[mask == 1]
    return f1_score(l, p, average=None, labels=[0, 1, 2])


def confusion_matrix_masked(preds, labels, mask):
    p = preds[mask == 1]
    l = labels[mask == 1]
    return confusion_matrix(l, p, labels=[0, 1, 2])