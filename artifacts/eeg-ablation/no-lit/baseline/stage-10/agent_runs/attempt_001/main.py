"""
This script runs a comparative study of static vs. dynamic graph neural networks for emotion recognition using EEG datasets.
Datasets: DEAP, SEED
Key Tasks:
1. Compare static predefined graphs (GCN) vs. dynamic graph construction (adaptive adjacency, temporal sliding-window).
2. Evaluate on the DEAP and SEED datasets.
3. Metrics: accuracy (primary), F1-score, training time, and community transition correlation (secondary).

HYPERPARAMETERS:
- learning_rate: 0.001
- batch_size: 32
- num_epochs: 50
- hidden_dim: 64
- max_window_size: 4 seconds
- min_window_size: 2 seconds
- community_detection_method: 'spectral_clustering'
"""

import numpy as np
import time
from experiment_harness import ExperimentHarness
from data_utils import load_dataset, preprocess_eeg, split_dataset
from models import StaticGraphGCN, DynamicEdgeGraphGAT, AdaptiveCommunityDGNN, SlidingWindowDGNN, HybridStaticDynamicGraph
from metrics import compute_accuracy, compute_f1_score

HYPERPARAMETERS = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 50,
    'hidden_dim': 64,
    'max_window_size': 4,
    'min_window_size': 2,
    'community_detection_method': 'spectral_clustering',
    'static_dynamic_weight_ratio': 0.5
}

SEEDS = [42, 123, 456]
MODELS = {
    "StaticGraphGCN": StaticGraphGCN,
    "DynamicEdgeGraphGAT": DynamicEdgeGraphGAT,
    "AdaptiveCommunityDGNN": AdaptiveCommunityDGNN,
    "SlidingWindowDGNN": SlidingWindowDGNN,
    "HybridStaticDynamicGraph": HybridStaticDynamicGraph,
}

DATASETS = ["DEAP", "SEED"]

def run_condition(model_class, dataset, seed):
    np.random.seed(seed)
    # Load and preprocess dataset
    data = load_dataset(dataset)
    train_data, val_data, test_data = split_dataset(data)
    train_data, val_data, test_data = preprocess_eeg(train_data), preprocess_eeg(val_data), preprocess_eeg(test_data)
    
    # Initialize model
    model = model_class(input_dim=train_data['features'].shape[1], 
                        hidden_dim=HYPERPARAMETERS['hidden_dim'], 
                        num_classes=train_data['labels'].max() + 1, 
                        **HYPERPARAMETERS)
    
    # Train and evaluate
    model.fit(train_data, val_data, epochs=HYPERPARAMETERS['num_epochs'])
    accuracy, f1_score = model.evaluate(test_data)
    return accuracy, f1_score

def main():
    harness = ExperimentHarness(time_budget=1800)
    print("REGISTERED_CONDITIONS:", ", ".join(MODELS.keys()))

    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        for model_name, model_class in MODELS.items():
            results[dataset][model_name] = []
            for seed in SEEDS:
                try:
                    accuracy, f1_score = run_condition(model_class, dataset, seed)
                    print(f"condition={model_name} dataset={dataset} seed={seed} accuracy: {accuracy:.4f}, f1_score: {f1_score:.4f}")
                    results[dataset][model_name].append((accuracy, f1_score))
                except Exception as e:
                    print(f"CONDITION_FAILED: {model_name} dataset={dataset} seed={seed} error={e}")
                if harness.should_stop():
                    break
    
    # Aggregate results
    for dataset, dataset_results in results.items():
        for model_name, metrics in dataset_results.items():
            accuracies = [m[0] for m in metrics]
            f1_scores = [m[1] for m in metrics]
            print(f"SUMMARY: dataset={dataset} condition={model_name} mean_accuracy: {np.mean(accuracies):.4f}, std_accuracy: {np.std(accuracies):.4f}, mean_f1: {np.mean(f1_scores):.4f}")
    
    harness.finalize()

if __name__ == "__main__":
    main()