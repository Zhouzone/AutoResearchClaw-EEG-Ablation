import numpy as np

def load_dataset(name):
    """
    Load synthetic EEG dataset based on the specified name.
    """
    np.random.seed(42)
    if name == "DEAP":
        features = np.random.rand(100, 32)
        labels = np.random.randint(0, 2, 100)
    elif name == "SEED":
        features = np.random.rand(150, 32)
        labels = np.random.randint(0, 3, 150)
    else:
        raise ValueError("Unknown dataset name.")
    return {'features': features, 'labels': labels}

def preprocess_eeg(data):
    """
    Standardize EEG features to have zero mean and unit variance.
    """
    data['features'] = (data['features'] - np.mean(data['features'], axis=0)) / np.std(data['features'], axis=0)
    return data

def split_dataset(data, seed=42):
    """
    Split dataset into train, validation, and test sets.
    """
    np.random.seed(seed)
    n = len(data['labels'])
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_idx = indices[:int(0.6 * n)]
    val_idx = indices[int(0.6 * n):int(0.8 * n)]
    test_idx = indices[int(0.8 * n):]
    return {
        'features': data['features'][train_idx],
        'labels': data['labels'][train_idx]
    }, {
        'features': data['features'][val_idx],
        'labels': data['labels'][val_idx]
    }, {
        'features': data['features'][test_idx],
        'labels': data['labels'][test_idx]
    }