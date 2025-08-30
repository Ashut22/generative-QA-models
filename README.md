# Question Answering with DistilBERT on Custom Medical QA Data  

##  Project Overview  
This project fine-tunes a **DistilBERT-based Question Answering (QA) model** on a custom dataset in **SQuAD format**.  
The dataset is derived from medical-style QA contexts, where the model is trained to predict answers to patient-related questions.  

The workflow covers:  
- Preprocessing **JSON data** into DataFrames.  
- Converting data into **SQuAD-compatible format**.  
- Fine-tuning **DistilBERT** using the **SimpleTransformers library**.  
- Evaluating performance with **Exact Match (EM)** and **F1 Score**.  

---

##  Tech Stack  
- **Python, Pandas, NumPy** → Data preprocessing  
- **PyTorch** → Model training backend  
- **Transformers (HuggingFace)** → Tokenization  
- **SimpleTransformers** → High-level QA model training  
- **Evaluate (HuggingFace)** → Computing SQuAD metrics  

---

##  Dataset  
- Input: `train.json` (SQuAD-like format with `context`, `question`, and `answers`)  
- Preprocessing: Extracts **question, context, and answer** into a Pandas DataFrame and converts them into SQuAD format.  

---

##  Training & Prediction  
1. Load and preprocess data (`train_data`, `test_data`).  
2. Train a **DistilBERT QA model** on training data.  
3. Generate predictions on test data using:  
   ```python
   predictions, raw_outputs = model.predict(test_data)
## Requirements
- Python 3.x
- Hugging Face Transformers
- TensorFlow (for GAN)
- PyTorch
- NumPy
- Pandas
- Matplotlib


