# SemantiQ: Bloom's Taxonomy Question Classifier

A machine learning system for automatically classifying educational questions by Bloom's Taxonomy cognitive levels using **SemantiQ** (Semantic-BERT and Semantic-FastText) approaches.

## 🎯 Overview

This project (SemantiQ) implements two complementary approaches for classifying educational questions into Bloom's Taxonomy cognitive levels:

| Level | Description | Example |
|-------|-------------|---------|
| **Remember** | Recall facts and basic concepts | "What is the capital of France?" |
| **Understand** | Explain ideas or concepts | "Describe how photosynthesis works." |
| **Apply** | Use information in new situations | "Calculate the area of a triangle." |
| **Analyze** | Draw connections among ideas | "Compare and contrast mitosis and meiosis." |
| **Evaluate** | Justify a decision or course of action | "Assess the effectiveness of this policy." |
| **Create** | Produce new or original work | "Design an experiment to test plant growth." |

## ✨ Key Features

- **Semantic Dependency Parsing**: Uses spaCy to extract word relationships (Subject-Verb-Object) for better intent detection
- **Merged Dataset**: 229 high-quality questions from EDUPRESS EP 729 and GitHub educational corpora
- **Dual Model Support**: Choose between lightweight FastText or powerful BERT
- **Streamlit Web App**: Interactive classification with probability visualizations
- **Pedagogically Aligned**: Results ordered by cognitive hierarchy

## 📁 Project Structure

```
SemantiQ/
├── data/
│   ├── raw/                    # Original dataset files (229 samples)
│   ├── processed/              # Cleaned and tokenized data
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── preprocessing.py        # Text cleaning + Semantic Dependency Parsing
│   ├── dataset.py              # PyTorch datasets
│   ├── fasttext_classifier.py  # FastText + SVM/MLP pipeline
│   ├── bert_classifier.py      # BERT fine-tuning
│   ├── evaluate.py             # Metrics and confusion matrices
│   └── inference.py            # Unified prediction API
├── models/
│   ├── fasttext/               # Saved FastText models
│   └── bert/                   # Saved BERT checkpoints
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_fasttext_train.ipynb # FastText training
│   └── 03_bert_train.ipynb     # BERT training
├── config/
│   └── config.yaml             # Hyperparameters
├── app.py                      # Streamlit Web Application
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### 2. Train FastText Model

```bash
python src/fasttext_classifier.py --data data/raw/bloom_questions.csv --text-col question --label-col level
```

### 3. Train BERT Model (requires GPU)

```bash
python src/bert_classifier.py --data data/raw/bloom_questions.csv --text-col question --label-col level --epochs 5
```

### 4. Run Streamlit App

```bash
streamlit run app.py
```

### 5. Run Inference (CLI)

```python
from src.inference import BloomClassifier

# FastText (faster, lighter)
classifier = BloomClassifier(model_type="fasttext")
result = classifier.predict("What is photosynthesis?")
print(result)
# {'level': 'Remember', 'confidence': 0.92, 'description': 'Recall facts and basic concepts'}

# BERT (higher accuracy)
classifier = BloomClassifier(model_type="bert")
result = classifier.predict("Design an experiment to test plant growth")
print(result)
# {'level': 'Create', 'confidence': 0.95, 'description': 'Produce new or original work'}
```

## 📊 Evaluation

Run evaluation on test set:

```bash
python src/evaluate.py --model both --test-data data/processed/test.csv
```

This generates:
- Accuracy, Precision, Recall, F1-score metrics
- Confusion matrices for each model
- Model comparison charts

## ⚙️ Configuration

Edit `config/config.yaml` to modify:

- Data split ratios
- FastText embedding parameters
- Classifier type (SVM or MLP)
- BERT fine-tuning hyperparameters
- **Semantic Parsing** (on/off)

## 📚 Methods (Based on Research Paper)

### Semantic-FastText (S-FastText)
1. **Semantic Dependency Parsing**: Extract word roles using spaCy
2. **Train FastText embeddings** on enriched text
3. **Generate sentence vectors** by averaging word embeddings
4. **Train SVM classifier** on sentence vectors

### Semantic-BERT (S-BERT)
1. **Load pre-trained** `bert-base-uncased` model
2. **Add classification head** (768 → 6 classes)
3. **Fine-tune end-to-end** with AdamW optimizer

## 📈 Dataset Information

| Source | Samples | Classes |
|--------|---------|---------|
| EDUPRESS EP 729 | ~100 | 6 |
| GitHub Educational Corpus | ~130 | 6 |
| **Total** | **229** | **6** |

### Class Distribution
- Create: 56 (24%)
- Understand: 51 (22%)
- Remember: 41 (18%)
- Analyze: 31 (14%)
- Evaluate: 30 (13%)
- Apply: 20 (9%)

## 🔬 Research Reference

This implementation is based on the paper:
> **"Semantic-BERT and semantic-FastText models for education question classification"**

Key contributions from the paper:
- Semantic Dependency Parsing for better intent detection
- Functional role enrichment of question text
- High accuracy on 5W1H educational questions

## 📄 License

MIT License
