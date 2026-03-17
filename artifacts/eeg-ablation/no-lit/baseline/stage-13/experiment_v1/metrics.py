import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_accuracy(predictions, labels):
    """
    Compute accuracy for multi-class predictions.
    """
    if len(predictions) == 0:
        return 0.0  # Edge case for empty predictions
    return accuracy_score(labels, predictions)

def compute_f1_score(predictions, labels, average='weighted'):
    """
    Compute F1-score using sklearn.
    Supports multi-class and weighted computation.
    """
    if len(predictions) == 0:
        return 0.0  # Edge case for empty predictions
    return f1_score(labels, predictions, average=average)