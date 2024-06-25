import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import emoji
import pickle

# Download NLTK resources
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Function to preprocess text
def text_preprocessing(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in emoji.UNICODE_EMOJI])
    text = ''.join([char for char in text if not char.isdigit()])
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    text = ' '.join(words)
    return text

# Function to load models
def load_models():
    models = {}
    with open('model_count.pkl', 'rb') as file:
        models['count'] = pickle.load(file)
    with open('model_word.pkl', 'rb') as file:
        models['word'] = pickle.load(file)
    with open('model_ngram.pkl', 'rb') as file:
        models['ngram'] = pickle.load(file)
    with open('model_chars.pkl', 'rb') as file:
        models['chars'] = pickle.load(file)
    return models

# Function to predict sentiment
def predict_sentiment(models, vectorizers, text):
    results = {}
    for key, model in models.items():
        vectorizer = vectorizers[key]
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)
        results[key] = 'Positive' if prediction[0] == 1 else 'Negative'
    return results

# Streamlit app
def main():
    st.title('Sentiment Analysis Demo')
    st.sidebar.title('User Input')

    # Load models and vectorizers
    models = load_models()
    vectorizers = {
        'count': CountVectorizer(vocabulary=models['count'].classes_),
        'word': TfidfVectorizer(vocabulary=models['word'].vocabulary_),
        'ngram': TfidfVectorizer(vocabulary=models['ngram'].vocabulary_, ngram_range=(2, 3)),
        'chars': TfidfVectorizer(vocabulary=models['chars'].vocabulary_, analyzer='char', ngram_range=(2, 3))
    }

    # User input
    user_input = st.sidebar.text_area('Enter your text here:')
    if st.sidebar.button('Predict'):
        if user_input.strip() != '':
            processed_text = text_preprocessing(user_input)
            st.write('Processed Text:', processed_text)
            st.write('Predictions:')
            results = predict_sentiment(models, vectorizers, processed_text)
            for key, result in results.items():
                st.write(f'{key.capitalize()} Model:', result)
        else:
            st.warning('Please enter some text.')

if __name__ == '__main__':
    main()
