import numpy as np
from data_utils import load_dataset, preprocess_eeg, split_dataset
from metrics import compute_accuracy, compute_f1_score
from models import StaticGraphGCN, DynamicEdgeGraphGAT, AdaptiveCommunityDGNN, SlidingWindowDGNN, HybridStaticDynamicGraph

def main():
    """
    Run experiments comparing different graph neural network models on EEG datasets.
    """
    np.random.seed(42)
    dataset_name = "SEED"  # Dataset to use
    seeds = range(5)  # Use 5 seeds for variability
    results = []

    models = {
        "StaticGraphGCN": StaticGraphGCN(input_dim=32, hidden_dim=64, num_classes=3, learning_rate=0.01),
        "DynamicEdgeGraphGAT": DynamicEdgeGraphGAT(input_dim=32, hidden_dim=64, num_classes=3, learning_rate=0.01),
        "AdaptiveCommunityDGNN": AdaptiveCommunityDGNN(input_dim=32, hidden_dim=64, num_classes=3, learning_rate=0.01),
        "SlidingWindowDGNN": SlidingWindowDGNN(input_dim=32, hidden_dim=64, num_classes=3, learning_rate=0.01),
        "HybridStaticDynamicGraph": HybridStaticDynamicGraph(input_dim=32, hidden_dim=64, num_classes=3, learning_rate=0.01),
    }

    for seed in seeds:
        print(f"Running experiment with seed {seed}...")
        data = load_dataset(dataset_name)
        data = preprocess_eeg(data)
        train_data, val_data, test_data = split_dataset(data, seed=seed)

        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"Training model: {model_name}")
            model.fit(train_data, val_data, epochs=50)
            predictions = model.predict(test_data['features'])
            acc = compute_accuracy(predictions, test_data['labels'])
            f1 = compute_f1_score(predictions, test_data['labels'], average='weighted')
            results.append({"condition": model_name, "seed": seed, "accuracy": acc, "f1_score": f1})

    # Summarize results
    print("\nSummary of results:")
    for model_name in models.keys():
        model_results = [r for r in results if r['condition'] == model_name]
        avg_acc = np.mean([r['accuracy'] for r in model_results])
        avg_f1 = np.mean([r['f1_score'] for r in model_results])
        print(f"condition={model_name} Avg Accuracy: {avg_acc:.4f}, Avg F1-Score: {avg_f1:.4f}")

if __name__ == "__main__":
    main()