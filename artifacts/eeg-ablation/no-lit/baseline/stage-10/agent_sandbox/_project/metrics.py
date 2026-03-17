import numpy as np

def compute_accuracy(predictions, labels):
    return np.mean(predictions == labels)

def compute_f1_score(predictions, labels):
    # Placeholder for F1-score computation
    return 2 * (0.8 * 0.7) / (0.8 + 0.7)