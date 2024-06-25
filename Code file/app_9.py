from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import emoji
import pickle

# Download NLTK resources
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Function to preprocess text
def text_preprocessing(text):
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    text = ' '.join(words)
    return text

# Function to load models
def load_models():
    models = {}
    with open('count_vectorizer.pkl', 'rb') as file:
        models['count_vectorizer'] = pickle.load(file)
    with open('rf_model_count.pkl', 'rb') as file:
        models['count_model'] = pickle.load(file)
    with open('tfidf_word_vectorizer.pkl', 'rb') as file:
        models['word_vectorizer'] = pickle.load(file)
    with open('rf_model_word.pkl', 'rb') as file:
        models['word_model'] = pickle.load(file)
    with open('tfidf_ngram_vectorizer.pkl', 'rb') as file:
        models['ngram_vectorizer'] = pickle.load(file)
    with open('rf_model_ngram.pkl', 'rb') as file:
        models['ngram_model'] = pickle.load(file)
    with open('tfidf_chars_vectorizer.pkl', 'rb') as file:
        models['chars_vectorizer'] = pickle.load(file)
    with open('rf_model_chars.pkl', 'rb') as file:
        models['chars_model'] = pickle.load(file)
    return models

# Function to predict sentiment
def predict_sentiment(models, text):
    results = {}
    
    # Count Vectorizer
    count_vectorizer = models['count_vectorizer']
    transformed_text_count = count_vectorizer.transform([text])
    result_count = models['count_model'].predict(transformed_text_count)
    results['count'] = 'Positive' if result_count[0] == 1 else 'Negative'
    
    # TF-IDF Word Vectorizer
    word_vectorizer = models['word_vectorizer']
    transformed_text_word = word_vectorizer.transform([text])
    result_word = models['word_model'].predict(transformed_text_word)
    results['word'] = 'Positive' if result_word[0] == 1 else 'Negative'
    
    # TF-IDF Ngram Vectorizer
    ngram_vectorizer = models['ngram_vectorizer']
    transformed_text_ngram = ngram_vectorizer.transform([text])
    result_ngram = models['ngram_model'].predict(transformed_text_ngram)
    results['ngram'] = 'Positive' if result_ngram[0] == 1 else 'Negative'
    
    # TF-IDF Chars Vectorizer
    chars_vectorizer = models['chars_vectorizer']
    transformed_text_chars = chars_vectorizer.transform([text])
    result_chars = models['chars_model'].predict(transformed_text_chars)
    results['chars'] = 'Positive' if result_chars[0] == 1 else 'Negative'
    
    return results

# Streamlit app
def main():
    st.title('Sentiment Analysis Demo')

    # Load models
    models = load_models()

    # User input
    user_input = st.text_area('Enter your text here:')
    if st.button('Predict'):
        if user_input.strip() != '':
            processed_text = text_preprocessing(user_input)
            st.write('Processed Text:', processed_text)
            st.write('Predictions:')
            results = predict_sentiment(models, processed_text)
            for key, result in results.items():
                st.write(f'{key.capitalize()} Model:', result)
        else:
            st.warning('Please enter some text.')

if __name__ == '__main__':
    main()
