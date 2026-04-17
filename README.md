# imdb-multimodal-genre-classification# Multimodal IMDB Film Genre Classification (CNN + LSTM)

## 📌 Overview

This project implements a multimodal deep learning approach to classify movie genres using:

* **CNN** for film poster images
* **LSTM** for movie overviews (text)

The task is a **multi-label classification problem**, where each film can belong to multiple genres.

---

## 🎯 Objectives

* Build GPU-accelerated data pipelines using TensorFlow
* Develop and train deep learning models in Keras
* Evaluate model performance and analyze results critically

---

## 🧠 Models

### 1. CNN (Image-based)

* Input: Movie posters
* Learns visual patterns such as color, layout, and objects
* Achieves good generalisation but low recall

### 2. LSTM (Text-based)

* Input: Movie overviews
* Learns semantic patterns from text
* Shows overfitting due to limited dataset

---

## 📊 Results Summary

| Model | Strengths                               | Weaknesses                        |
| ----- | --------------------------------------- | --------------------------------- |
| CNN   | Good precision (~0.67), stable training | Low recall, biased toward "Drama" |
| LSTM  | Captures keyword signals                | Overfitting, poor generalisation  |

---

## 🔍 Key Observations

* Strong **class imbalance** (dominance of "Drama")
* CNN struggles with visually similar genres
* LSTM depends heavily on keyword presence
* Overfitting observed in LSTM after ~5 epochs

---

## 🚀 Improvements

* Use **transfer learning** (ResNet, EfficientNet)
* Apply **pre-trained embeddings** (GloVe, FastText)
* Handle class imbalance (class weights / oversampling)
* Combine both models for **multimodal learning**

---

## 📁 Files

* `Keras_Assignment.ipynb` → Model implementation
* `report.pdf` → Critical analysis
* `images/` → Training plots and predictions

---

## ⚙️ Requirements

* Python 3.x
* TensorFlow / Keras
* NumPy, Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 👤 Author

Laxmi Narayana Gugulothu

---

## 📌 Notes

This project was developed as part of a coursework assignment on GPU-accelerated deep learning using TensorFlow.
