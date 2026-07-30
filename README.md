# 🧠 NLP Sentiment Analysis with Word2Vec & RNN

This repository contains a complete Deep Learning pipeline for Natural Language Processing (NLP). It demonstrates how to classify text (Sentiment Analysis) using custom **Word2Vec embeddings** and a **Recurrent Neural Network (RNN)** built with Keras/TensorFlow.

## 🚀 Pipeline Architecture

1. **Text Preprocessing:** 
   - Tokenization of raw text using Keras `Tokenizer`.
   - Sequence padding to ensure uniform input dimensions for the neural network.
   - Label encoding for binary classification (Positive/Negative).

2. **Word Embedding (Word2Vec):** 
   - Training a custom `Word2Vec` model using Gensim to map words into a 50-dimensional semantic vector space.
   - Creating a frozen embedding matrix to feed pre-trained semantic context into the neural network.

3. **Deep Learning Model (RNN):**
   - **Embedding Layer:** Loaded with custom Word2Vec weights (`trainable=False`).
   - **SimpleRNN Layer:** Processes text sequentially to capture contextual dependencies.
   - **Dense Layer:** Uses a `sigmoid` activation function to output a probability score (Binary Crossentropy).

## 🛠️ Tech Stack
- `Python`
- `TensorFlow / Keras` (Model building and training)
- `Gensim` (Word2Vec generation)
- `Scikit-learn` (Train/Test splitting, Label Encoding)
- `Numpy` & `Pandas` (Data manipulation)

## 💻 Example Usage

The project includes a custom inference function to test new, unseen sentences:

```python
sentence = "Restaurant çok temizdi ve yemekler çok güzeldi"
result = classify_sentence(sentence)
print(f"Result: {result}")
# Output: Result: positive
