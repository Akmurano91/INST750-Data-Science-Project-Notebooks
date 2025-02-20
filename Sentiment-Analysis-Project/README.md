""# ChatGPT Reviews Sentiment Analysis

## Overview
This project analyzes user reviews of the ChatGPT Android App to classify sentiment as positive or negative. It utilizes both traditional machine learning models (Logistic Regression) and a deep learning-based approach (RoBERTa) to perform sentiment classification.

## Installation
To run this project, install the required dependencies:
```bash
pip install transformers datasets accelerate nltk scikit-learn pandas matplotlib seaborn torch
```

## Dataset
The dataset consists of over **381,000** user reviews, including columns such as:
- `reviewId` - Unique identifier for the review.
- `userName` - Name of the reviewer.
- `content` - The text of the review.
- `score` - Rating provided by the user (1-5).
- `thumbsUpCount` - Number of users who found the review helpful.
- `reviewCreatedVersion` - The app version at the time of the review.
- `date` - The date the review was posted.
- `appVersion` - The version of the ChatGPT app.

## Methodology
### Data Preprocessing
1. **Cleaning**: Removal of missing values in the `content` and `score` columns.
2. **Text Processing**:
   - Lowercasing
   - Removal of non-alphabetic characters
   - Stopword removal using NLTK
3. **Label Encoding**:
   - Reviews with scores **> 3** are labeled as **positive (1)**.
   - Reviews with scores **≤ 3** are labeled as **negative (0)**.

### Train/Test Split
The dataset is split into **80% training** and **20% testing**.

## Models & Experiments
### Traditional Machine Learning Model
- **Logistic Regression** trained on **TF-IDF** vectorized text (2000 features).
- **Accuracy: 91.46%**

### RoBERTa Transformer Model
- **Pretrained `roberta-base` model** fine-tuned for sentiment classification.
- **Optimizations**:
  - Dataset subset to **2% (1/50th)** for training and testing.
  - **First six layers frozen** to reduce training parameters.
  - **Learning rate: 2e-5**, **Batch size: 8**, **Epochs: 2**.
- **Training conducted using Hugging Face's Trainer API.**
- **Final Accuracy: 89.79%**

## Usage
Run preprocessing and train the models:
```bash
python train.py
```

Evaluate model performance:
```bash
python evaluate.py
```

## References
- Google Colab - Gemini
- Hugging Face RoBERTa Documentation
- DataCamp - Working with Hugging Face
- DataCamp - AI Engineer Career Track
- CodeSignal - TF-IDF Vectorization
- Towards Data Science - NLP
