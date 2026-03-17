# Research Proposal: Temporal-Adaptive Graph Networks for EEG-based Emotion Recognition

## Problem Statement

Emotion recognition from EEG signals is challenged by the dynamic and time-varying nature of functional brain connectivity. Current approaches relying on static graph structures (e.g., adjacency matrices based on electrode locations or fixed priors) fail to capture the temporal evolution of connectivity patterns and individual variability in emotional states. This leads to suboptimal generalization, especially in cross-subject and small-sample scenarios.

We propose **Temporal-Adaptive Graph Networks (TAGN)**, a novel framework that dynamically learns time-varying adjacency matrices to model evolving functional brain connectivity. TAGN integrates attention-driven graph learning, sparsity and smoothness regularization, and a meta-learning strategy to generalize across subjects.

---

## Key Contribution

**Dynamic Adjacency Matrix Learning Framework**: We introduce a self-supervised, attention-driven mechanism to generate dynamic adjacency matrices for modeling time-varying connectivity. This mechanism:
- Utilizes multi-head self-attention to compute connectivity weights between EEG channels for each time window.
- Incorporates sparsity and temporal smoothness regularization to constrain complexity and mitigate overfitting.
- Adapts to individual variability by leveraging meta-learning to extract shared graph priors across subjects.

This framework is designed to outperform static graph-based methods in capturing dynamic, sparse, and interpretable connectivity patterns.

---

## Technical Approach

### 1. Architecture Overview
The proposed TAGN architecture consists of three key modules:

#### (a) **Dynamic Graph Structure Learning (DGSL)**
- **Input**: Preprocessed multi-channel EEG signals, segmented into time windows of fixed length.
- **Mechanism**: For each time window:
  - Compute feature embeddings for EEG channels using a shared temporal convolutional encoder.
  - Generate a dynamic adjacency matrix through a multi-head self-attention mechanism:
    \[
    A_t = \text{Softmax}\left(\frac{Q_t K_t^\top}{\sqrt{d_k}}\right),
    \]
    where \( Q_t \) and \( K_t \) are query and key vectors derived from channel embeddings at time \( t \), and \( d_k \) is the embedding dimension.
  - Apply sparsity regularization (\( L_1 \) norm) to enforce fewer, meaningful edges:
    \[
    \mathcal{L}_{\text{sparsity}} = \lambda_1 \|A_t\|_1.
    \]
  - Incorporate temporal smoothness across consecutive adjacency matrices:
    \[
    \mathcal{L}_{\text{smoothness}} = \lambda_2 \sum_t \|A_t - A_{t-1}\|_F^2.
    \]

#### (b) **Spatial-Temporal Feature Extraction**
- **Graph Convolution**: Perform spatial feature aggregation using Graph Convolutional Networks (GCN) on the dynamic adjacency matrices \( A_t \) and EEG channel features.
- **Temporal Modeling**: Use a Gated Recurrent Unit (GRU) to capture temporal dependencies among aggregated spatial features across time windows.

#### (c) **Meta-Learning for Cross-Subject Generalization**
- **Framework**: Use Model-Agnostic Meta-Learning (MAML) to improve cross-subject generalization:
  - **Inner Loop**: Train TAGN on individual subject data to adapt adjacency matrix generation and classification layers.
  - **Outer Loop**: Optimize shared initialization parameters across subjects to balance generalization and individual variability.
- This enables the model to quickly adapt to new subjects with limited data.

---

### 2. Loss Function
The total loss function combines classification loss, sparsity regularization, and smoothness regularization:
\[
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_1 \|A_t\|_1 + \lambda_2 \sum_t \|A_t - A_{t-1}\|_F^2,
\]
where \( \mathcal{L}_{\text{CE}} \) is the cross-entropy loss for emotion classification.

---

## Experimental Plan

### 1. Datasets and Preprocessing
- **Datasets**: Use publicly available EEG datasets:
  - **SEED**: 62 channels, 3 emotion categories (positive, neutral, negative).
  - **DEAP**: 32 channels, valence/arousal ratings (binary classification).
- **Preprocessing**: Bandpass filtering (0.5–50 Hz), z-score normalization, segment signals into 1-second non-overlapping windows.

### 2. Baselines
Compare TAGN against:
- **Static Graph Models**: GCNN with fixed adjacency matrices (e.g., based on electrode locations).
- **Dynamic Graph Models**: Temporal Graph Convolutional Networks (T-GCN).
- **Non-graph Models**: LSTM, CNN.

### 3. Evaluation Metrics
- Classification accuracy and F1-score for emotion recognition.
- Cross-subject generalization performance (leave-one-subject-out validation).
- Ablation experiments to validate the contributions of:
  - Dynamic adjacency matrix (replace with static adjacency).
  - Sparsity (\( \lambda_1 = 0 \)).
  - Smoothness (\( \lambda_2 = 0 \)).

### 4. Interpretability Analysis
- Visualize learned adjacency matrices for different emotion classes to identify significant connectivity patterns.
- Compare these patterns with known emotion-related brain networks (e.g., frontal-temporal connectivity).

---

## Expected Outcomes
1. **Performance Gains**: Achieve 3–5% higher cross-subject classification accuracy compared to static graph-based methods.
2. **Robustness**: Demonstrate reduced overfitting in small-sample scenarios through sparsity and smoothness regularization.
3. **Interpretability**: Provide neurophysiological insights by visualizing emotion-specific dynamic brain connectivity.

---

## Conclusion
This research addresses the limitations of static graph methods in EEG-based emotion recognition by introducing a robust and interpretable framework for dynamic graph learning. TAGN balances model flexibility, generalization, and interpretability, setting a new standard for emotion recognition tasks under dynamic and cross-subject conditions.