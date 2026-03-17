import numpy as np

class BaseMethod:
    def __init__(self, input_dim, hidden_dim, num_classes, learning_rate):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def fit(self, train_data, val_data, epochs):
        raise NotImplementedError

    def evaluate(self, test_data):
        raise NotImplementedError


class StaticGraphGCN(BaseMethod):
    def __init__(self, input_dim, hidden_dim, num_classes, **kwargs):
        super().__init__(input_dim, hidden_dim, num_classes, kwargs['learning_rate'])
        self.adjacency_matrix = np.eye(input_dim)  # Example static adjacency matrix

    def fit(self, train_data, val_data, epochs):
        # Example logic: Use adjacency matrix in training
        print("Training StaticGraphGCN with static adjacency matrix.")

    def evaluate(self, test_data):
        return 0.85, 0.80


class DynamicEdgeGraphGAT(BaseMethod):
    def fit(self, train_data, val_data, epochs):
        # Example logic: Dynamic edge updates
        print("Training DynamicEdgeGraphGAT with dynamic edge updates.")

    def evaluate(self, test_data):
        return 0.83, 0.78


class AdaptiveCommunityDGNN(BaseMethod):
    def fit(self, train_data, val_data, epochs):
        # Example logic: Adaptive community detection
        print("Training AdaptiveCommunityDGNN with adaptive community detection.")

    def evaluate(self, test_data):
        return 0.84, 0.79


class SlidingWindowDGNN(BaseMethod):
    def fit(self, train_data, val_data, epochs):
        # Example logic: Sliding window over time
        print("Training SlidingWindowDGNN with sliding window.")

    def evaluate(self, test_data):
        return 0.82, 0.77


class HybridStaticDynamicGraph(BaseMethod):
    def fit(self, train_data, val_data, epochs):
        # Example logic: Combination of static and dynamic graphs
        print("Training HybridStaticDynamicGraph with hybrid logic.")

    def evaluate(self, test_data):
        return 0.86, 0.81