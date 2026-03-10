# 🎬 Sentiment Analysis — Bidirectional LSTM

> Classifies IMDb movie reviews as positive or negative using a Bidirectional LSTM neural network.

---

## Overview

This project builds a deep learning NLP pipeline on the IMDb Movie Reviews dataset (50,000 reviews). A Bidirectional LSTM is used to capture context from both directions of a review sequence, improving understanding of language nuance. The model is regularized with dropout and early stopping, and evaluated with a full classification report and confusion matrix.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

---

## Model Architecture

```
Embedding(vocab=10000, output=128, input_length=200)
  └── Bidirectional LSTM(64)
  └── Dropout(0.5)
  └── Dense(64, relu)
  └── Dense(1, sigmoid)
```

---

## How to Run

```bash
git clone https://github.com/Khalfani04/sentiment-analysis-lstm
cd sentiment-analysis-lstm
pip install tensorflow numpy seaborn scikit-learn matplotlib
jupyter notebook Assignment_4_COSC_31000.ipynb
```

---

## Training Details

| Setting | Value |
|---------|-------|
| Dataset | IMDb (keras.datasets.imdb) |
| Vocab Size | 10,000 most frequent words |
| Max Sequence Length | 200 tokens |
| Batch Size | 128 |
| Epochs | 5 (with early stopping) |
| Regularization | Dropout + EarlyStopping |

---

## Results

- Strong test accuracy on held-out IMDb reviews
- Bidirectional context improved classification over standard LSTM
- Evaluated with confusion matrix and full classification report (precision, recall, F1)

---

## What I Learned

- How Bidirectional LSTMs capture forward and backward sequence context
- How word embeddings represent semantic relationships between words numerically
- How early stopping and dropout work together to reduce overfitting in sequence models

---

## Contact

**Khalfani Norman** · [LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN) · [GitHub](https://github.com/Khalfani04)
