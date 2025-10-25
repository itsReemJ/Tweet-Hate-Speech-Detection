# tweet-hate-speech-detection
A Natural Language Processing group project in my junior year focusing on detecting hate speech in tweets using GRU and BERT models.

## Project Overview
This project aims to build an AI system that can automatically detect hateful or offensive content in tweets.  
It combines traditional NLP preprocessing with deep learning techniques to classify tweets as **hate speech** or **non-hate speech**.

The pipeline includes:
- Text cleaning and preprocessing (removing URLs, mentions, punctuation)
- Tokenization and lemmatization
- POS tagging and n-gram feature generation
- Word embeddings using **GloVe**
- Model training using **GRU** and **BERT**
- Model evaluation using accuracy, precision, recall, and F1-score

---

## Models Used
- **GRU (Gated Recurrent Unit):**  
  A type of recurrent neural network that captures sequential dependencies between words using GloVe embeddings.

- **BERT (Bidirectional Encoder Representations from Transformers):**  
  A transformer-based model fine-tuned on tweet data to capture deep contextual meanings in text.

---

## Technologies Used
- **Python**
- **NLTK** and **SpaCy** – for preprocessing and POS tagging  
- **Scikit-learn** – for feature extraction and metrics  
- **TensorFlow / PyTorch** – for deep learning models  
- **Hugging Face Transformers** – for BERT implementation  
- **Matplotlib / Seaborn** – for visualization  


## Results
| Model | Accuracy | F1-Score |
|--------|-----------|----------|
| GRU | 0.91 | 0.75 |
| BERT | 0.88 | 0.68 |

---

## Dataset
The dataset used in this project is the Hate Speech and Offensive Language Dataset from Kaggle. It contains tweets annotated for hate speech, offensive language, and neither-offensive.

Dataset Link: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
