# Siamese Network for Face Verification (Zero-Shot Learning)
A production-ready implementation of a Siamese Neural Network optimized for face verification using Metric Learning. The project leverages a pre-trained ResNet-18 backbone combined with custom L2 Embedding Normalization and a dynamic Triplet Margin Loss sampling pipeline.

This system achieves Zero-Shot / Few-Shot Learning capabilities, meaning it can verify whether two faces belong to the same person without ever being explicitly trained on those specific identities.

🚀 Key Engineering Highlights
* Custom Triplet Data Pipeline: Built from scratch to dynamically sample (Anchor, Positive, Negative) image batches in real-time.

* Data Leak Fix (Memory Optimization): Upgraded the pipeline to incorporate over 4,000 single-image identities from the LFW dataset as tough negative samples instead of discarding them, maximizing structural diversity.

* Hardware-Accelerated (Apple Silicon): Fully integrated with PyTorch's mps backend for low-latency training on Apple Silicon (M-series) chips.

* Automated Early Stopping: Monitored training loop featuring a deepcopy rollback mechanism that stops training when the loss plateaus, preventing overfitting.

* Production UI (Gradio Web App): A modern, drag-and-drop web application interface running locally to test face verification in real-time.

* Configuration Separation (MLOps): Completely isolated project hyperparameters and paths inside an external baseline.yaml file.

## Structure
```text
siamese_project/
│
├── configs/
│   └── baseline.yaml
│
├── data/
│   └── lfw/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── eval.py
│
├── app.py
├── main.py
└── best_siamese_model.pth
```
📊 Technical Architecture & Mathematics
The Objective
Instead of predicting class labels, the network maps high-dimensional facial images into a low-dimensional vector space ($\mathbb{R}^{128}$) where the geometric distance represents facial dissimilarity.

Embedding Normalization
The model applies L2 Normalization to the final output layer:
$$\text{Embedding} = \frac{f(x)}{\|f(x)\|_2}$$
This forces all output vectors to lie on a unit hypersphere, stabilizing the Euclidean distance scale and rendering a default decision threshold of 0.8 highly effective.

Loss Function
The model optimizes weights using Triplet Margin Loss, pushing matching pairs closer together while forcing non-matching faces apart by at least a predefined safety margin ($m$):
$$\mathcal{L}(A, P, N) = \max \left( d(A, P) - d(A, N) + \text{margin}, 0 \right)$$