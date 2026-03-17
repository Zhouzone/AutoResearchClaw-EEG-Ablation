import numpy as np

def load_dataset(name):
    # Placeholder for dataset loading.
    return {'features': np.random.rand(100, 32), 'labels': np.random.randint(0, 2, 100)}

def preprocess_eeg(data):
    # Placeholder for EEG preprocessing
    data['features'] = (data['features'] - np.mean(data['features'], axis=0)) / np.std(data['features'], axis=0)
    return data

def split_dataset(data):
    n = len(data['labels'])
    return {
        'features': data['features'][:int(0.6 * n)],
        'labels': data['labels'][:int(0.6 * n)]
    }, {
        'features': data['features'][int(0.6 * n):int(0.8 * n)],
        'labels': data['labels'][int(0.6 * n):int(0.8 * n)]
    }, {
        'features': data['features'][int(0.8 * n):],
        'labels': data['labels'][int(0.8 * n):]
    }