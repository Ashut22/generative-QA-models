# generative-QA-models
# Generative QA Model for Medical Transcription Analysis

This project focuses on implementing Generative Question Answering (QA) models to accurately extract specific patient details from medical transcriptions, ensuring high exact match rates and robust performance evaluation.

## Features

- **Automated Information Extraction**: Extracts key patient details, including:
  - Age
  - Complaints
  - Gender
  - Reasons for consultation
  - Symptoms
- **High-Accuracy Models**: Leverages pretrained models like **BioBERT** and **BERT-Large**, fine-tuned on medical QA datasets.
- **F1 Score Evaluation**: Employs token-level and field-specific F1 scores to assess accuracy.
- **Real-World Dataset**: Utilizes detailed medical transcription data to ensure practical applicability.

---

## Key Objectives

### 1. Learn Generative AI Fundamentals
- Understand concepts like tokenization, transformers, and fine-tuning.
- Explore the application of pretrained models to domain-specific problems.

### 2. Develop and Evaluate QA Models
- Fine-tune models like **BioBERT** and **BERT** for extracting structured patient information.
- Assess performance using token-level and field-specific F1 scores.

### 3. Gain Practical Experience
- Work with real-world datasets containing medical transcription data.
- Debug and optimize pipelines for medical QA tasks.

---

## Project Components

### Model Training and Inference
- Fine-tuned **BioBERT** and **BERT** models tailored for QA tasks.
- Implementation of a **Generative QA pipeline** for structured information extraction.

### Evaluation
- Calculation of **field-specific F1 scores** to assess accuracy.
- Computation of **average F1 scores** across multiple samples.

### Hands-On Learning
- A structured learning approach to ensure a comprehensive understanding of generative AI.
- Practical implementation using real-world medical transcription datasets.

---

## Getting Started

### Prerequisites
- **Python**: 3.8+
- Required Libraries: 
  - `transformers`
  - `datasets`
  - `tensorflow`
  - `seqeval`
  - `numpy`
  - `pandas`

