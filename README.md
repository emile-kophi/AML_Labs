# Deep Learning Applications - Repository Overview

This repository collects a series of **deep learning laboratory exercises** covering neural network architectures, transformers, and adversarial learning. Each lab is **self-contained**, includes modular code, and leverages **Weights & Biases** for experiment tracking and visualization. The repository is designed to provide a hands-on, practical experience in implementing, training, and evaluating deep learning models.

---

## **Laboratories Overview**

### **Laboratory 1 — Neural Network Architectures and Residual Connections**
- **Objective:** Investigate the impact of residual connections in fully connected and convolutional neural networks.
- **Key Features:**
  - Modular implementations of MLPs and CNNs with configurable residual blocks
  - Custom training pipeline with **early stopping** based on stagnation and overfitting
  - Dataset preprocessing for MNIST, CIFAR-10, and CIFAR-100
  - Experiment tracking and visualization using Weights & Biases
- **Experiments:**
  - Baseline MLP on MNIST
  - Residual MLP comparison
  - CNN vs Residual CNN on CIFAR-10
  - Fine-tuning pre-trained Residual CNN on CIFAR-100
- **Highlights:** Residual connections improved **gradient flow**, convergence speed, and final accuracy. Fine-tuning pre-trained networks outperformed classical classifiers trained on frozen features.

> See the [Laboratory 1 README](./Exercise_1/README.md) for full implementation and results.

---

### **Laboratory 3 — Transformers in the HuggingFace Ecosystem**
- **Objective:** Adapt pre-trained transformers to a text classification task and explore HuggingFace workflows.
- **Key Features:**
  - Feature extraction with **DistilBERT embeddings** for SVM baseline
  - Full fine-tuning of DistilBERT for sentiment classification on Rotten Tomatoes
  - Parameter-efficient fine-tuning with **LoRA (Low-Rank Adaptation)**
  - Evaluation metrics: accuracy, precision, recall, and F1-score
- **Highlights:** 
  - Fine-tuning DistilBERT improved performance compared to the SVM baseline.
  - LoRA reduced memory footprint while maintaining competitive accuracy.
  - Demonstrates HuggingFace’s flexibility from feature extraction to full fine-tuning.

> See the [Laboratory 3 README](./Exercise_3/README.md) for detailed methodology and results.

---

### **Laboratory 3 — Adversarial Learning**
- **Objective:** Study the vulnerability of neural networks to adversarial attacks and enhance model robustness.
- **Key Features:**
  - Evaluation of **untargeted and targeted FGSM attacks**
  - Modular FGSM function for both attack modes
  - Integration with Weights & Biases for monitoring attack effectiveness
  - On-the-fly adversarial training to improve robustness
- **Experiments:**
  - OOD detection using Maximum Softmax Probability (MSP)
  - Generation of adversarial examples with FGSM
  - Robustness improvement via adversarial training
  - Targeted FGSM attacks and quantitative success analysis
- **Highlights:** 
  - Neural networks are highly sensitive to small perturbations.
  - Adversarial training improves robustness but does not fully eliminate vulnerabilities.
  - Targeted attacks demonstrate controllable manipulation of model predictions.

> See the [Laboratory 4 README](./Exercise_4/README.md) for full details, experiments, and results.

---

## **Repository Structure**

```
AML_Labs/
│
├── Exercise_1/ # Neural networks, residual connections, CIFAR/MNIST experiments
│ ├── README.md
│ ├── images/ 
│ └── models/ 
│
├── Exercise_3/ # Transformers and HuggingFace experiments
│ ├── README.md
│ └── images/ 
│
├── Exercise_4/ # Adversarial learning and FGSM attacks
│ ├── README.md
│ ├── images/ 
│ └── models/ 
│
└── utils/ # Helper functions, model definitions, training utilities         
```
