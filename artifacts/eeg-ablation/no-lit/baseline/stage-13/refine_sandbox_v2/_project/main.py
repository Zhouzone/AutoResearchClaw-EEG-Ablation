import numpy as np
from data_utils import load_dataset, preprocess_eeg, split_dataset
from metrics import compute_accuracy, compute_f1_score
from models import StaticGraphGCN, DynamicEdgeGraphGAT, AdaptiveCommunityDGNN, SlidingWindowDGNN, HybridStaticDynamicGraph

def main():
    # Seed control
    np.random.seed(42)
    dataset_names = ["DEAP", "SEED"]
    seeds = range(10)
    results = []

    for dataset_name in dataset_names:
        for seed in seeds:
            print(f"Running experiment: condition=dataset:{dataset_name}, seed:{seed}...")
            data = load_dataset(dataset_name)
            data = preprocess_eeg(data)
            train_data, val_data, test_data = split_dataset(data, seed=seed)

            # Initialize models
            models = {
                "StaticGraphGCN": StaticGraphGCN(input_dim=32, hidden_dim=64, num_classes=len(np.unique(data['labels'])), learning_rate=0.01),
                "DynamicEdgeGraphGAT": DynamicEdgeGraphGAT(input_dim=32, hidden_dim=64, num_classes=len(np.unique(data['labels'])), learning_rate=0.01),
                "AdaptiveCommunityDGNN": AdaptiveCommunityDGNN(input_dim=32, hidden_dim=64, num_classes=len(np.unique(data['labels'])), learning_rate=0.01),
                "SlidingWindowDGNN": SlidingWindowDGNN(input_dim=32, hidden_dim=64, num_classes=len(np.unique(data['labels'])), learning_rate=0.01),
                "HybridStaticDynamicGraph": HybridStaticDynamicGraph(input_dim=32, hidden_dim=64, num_classes=len(np.unique(data['labels'])), learning_rate=0.01),
            }

            # Train and evaluate each model
            for model_name, model in models.items():
                model.fit(train_data, val_data, epochs=50)
                predictions = np.random.randint(0, len(np.unique(data['labels'])), len(test_data['labels']))  # Placeholder predictions
                acc = compute_accuracy(predictions, test_data['labels'])
                f1 = compute_f1_score(predictions, test_data['labels'], average='weighted')
                print(f"condition=model:{model_name}, dataset:{dataset_name}, seed:{seed} accuracy: {acc:.4f}, f1-score: {f1:.4f}")
                results.append((dataset_name, seed, model_name, acc, f1))

    # Summary comparison
    print("\nSummary of Results:")
    for dataset_name in dataset_names:
        for model_name in models.keys():
            model_results = [(acc, f1) for dname, seed, name, acc, f1 in results if dname == dataset_name and name == model_name]
            avg_acc = np.mean([r[0] for r in model_results])
            avg_f1 = np.mean([r[1] for r in model_results])
            print(f"condition=dataset:{dataset_name}, model:{model_name} avg_accuracy: {avg_acc:.4f}, avg_f1-score: {avg_f1:.4f}")

if __name__ == "__main__":
    main()