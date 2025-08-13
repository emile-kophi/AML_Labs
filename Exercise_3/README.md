# **Working with Transformers in the HuggingFace Ecosystem - Laboratory 3**

## **Overview**

This laboratory exercise explores how to work with the HuggingFace ecosystem to adapt pre-trained transformer models to new tasks.  
The main focus is on understanding the inner workings of HuggingFace abstractions and building a reproducible adaptation pipeline for text classification.  

Throughout the exercises, students will:  
- Download, explore, and preprocess datasets using the HuggingFace `datasets` library.  
- Use pre-trained transformer models, specifically **DistilBERT**, for feature extraction and fine-tuning.  
- Build stable baselines with classical classifiers (e.g., SVM) using transformer embeddings.  
- Fine-tune models for sequence classification tasks with HuggingFace `Trainer` and `TrainingArguments`.  
- Explore efficient fine-tuning techniques, including parameter-efficient methods using HuggingFace PEFT.  

The laboratory emphasizes hands-on experimentation, incremental pipeline development, and careful evaluation of model performance.

## **Introduction**

Transformers have become the backbone of modern natural language processing, but working with them can be challenging due to their complexity.  
The HuggingFace ecosystem provides high-level abstractions that simplify tasks such as feature extraction, fine-tuning, and model deployment.  

This laboratory is structured in progressive exercises:

1. **Sentiment Analysis Warm-up**  
   - Students start from a pre-trained DistilBERT model and use it as a feature extractor.  
   - The Rotten Tomatoes dataset is loaded and explored to verify splits, labels, and sample sentences.  
   - DistilBERT embeddings are extracted from the last hidden layer for selected samples.  
   - A baseline classifier (e.g., SVM) is trained on these embeddings to establish a stable reference point for further experiments.  

2. **Fine-tuning DistilBERT**  
   - The model is prepared for sequence classification by adding a classification head on top of the `[CLS]` token representation.  
   - Dataset splits are tokenized, and input IDs and attention masks are generated for training.  
   - A HuggingFace `Trainer` is configured with data collation, evaluation metrics, and training arguments.  
   - Students learn to perform full fine-tuning on the training split and evaluate performance on validation and test sets.  

3. **Efficient Fine-tuning Techniques**  
   - Explores methods to reduce the computational cost of fine-tuning, such as freezing layers or using parameter-efficient approaches from the HuggingFace **PEFT library**.  
   - Emphasis is on maintaining performance while minimizing resource usage and training time.  

## **Key Insights from Exercises 1.1, 1.2, and 1.3**

- **Dataset Exploration**:  
  - Verified available splits (`train`, `validation`, `test`) and label distribution.  
  - Inspected sample sentences to understand dataset structure and content.  

- **Feature Extraction with DistilBERT**:  
  - Loaded pre-trained DistilBERT and tokenizer.  
  - Tokenized sample texts and performed forward passes to extract **last hidden states**.  
  - Checked the position of the `[CLS]` token to ensure embeddings are extracted from the correct position.  

- **Baseline Classifier (Exercise 1.3)**:  
  - Used the `[CLS]` token embeddings as input features for a **linear SVM classifier**.  
  - Trained the SVM on the training split and evaluated on validation and test sets.  
  - Achieved reasonable baseline performance, confirming that DistilBERT embeddings capture meaningful semantic information even before fine-tuning.  

- **Initial Observations**:  
  - Pre-trained transformer embeddings provide strong representations for downstream classification tasks.  
  - Verifying token positions and extracting the correct embeddings is crucial for stable baseline construction.  
  - Baseline classifiers allow quick evaluation and comparison before committing to full fine-tuning.  

By the end of this laboratory, students will be able to take any pre-trained transformer model and adapt it to a specific NLP task, while understanding the key abstractions, tokenization strategies, and design choices within HuggingFace workflows.

## **Exercise 2 — Fine-tuning DistilBERT**

In this exercise, the pre-trained DistilBERT model is adapted to the Rotten Tomatoes sentiment analysis dataset for **sequence classification**.  

### **Tokenization and Dataset Preparation**

- All dataset splits (`train`, `validation`, `test`) are tokenized using the DistilBERT tokenizer.  
- Padding and truncation are applied to ensure uniform sequence length.  
- Tokenized datasets are formatted to be compatible with PyTorch (`input_ids`, `attention_mask`, `label`).  
- Number of output classes is set to 2 (positive vs. negative sentiment).

### **Model Setup**

- DistilBERT is instantiated with a **sequence classification head**, initialized randomly for the classification task.  
- The model is configured for **fine-tuning**, allowing gradients to update all layers.  

### **Training Pipeline**

- HuggingFace `Trainer` is used to manage the training loop.  
- A `DataCollatorWithPadding` ensures proper dynamic batch padding.  
- Evaluation metrics include **accuracy, precision, recall, and F1 score** (weighted).  
- Training is logged to **Weights & Biases (WandB)** for real-time monitoring.  
- Training arguments include learning rate, batch size, number of epochs, weight decay, logging frequency, and checkpoint saving.  
- Early stopping is implicitly handled by saving the best model at the end of training based on validation accuracy.  

### **Evaluation and Results**

- The fine-tuned DistilBERT model is evaluated on the validation and test splits.  
- Comparison is made with the **SVM baseline** trained on DistilBERT embeddings (Exercise 1.3).  
- Fine-tuning yields **higher accuracy**, demonstrating the advantage of updating model weights for the specific classification task.  

**Example Results (Test Set Accuracy):**

| Model                           | Accuracy |
|---------------------------------|----------|
| SVM (feature extractor)         | `svm_test_acc` |
| Fine-tuned DistilBERT           | `ft_test_acc`  |

- A visual comparison highlights the improvement of the fine-tuned model over the baseline SVM classifier.  
- Fine-tuning not only leverages pre-trained embeddings but also adapts the representation to the dataset-specific task, improving overall performance.  

**Key Takeaways:**

- Pre-trained transformers provide strong embeddings, but task-specific fine-tuning significantly enhances classification performance.  
- HuggingFace `Trainer` simplifies the setup of a complete fine-tuning pipeline including tokenization, batching, evaluation, and logging.  
- Maintaining proper dataset preparation and metric computation is critical for reliable evaluation.
