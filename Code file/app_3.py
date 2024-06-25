import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import pickle
import emoji
import nltk

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Function to load models
@st.cache(allow_output_mutation=True)
def load_models():
    with open('model_count.pkl', 'rb') as file:
        model_count = pickle.load(file)
    with open('model_word.pkl', 'rb') as file:
        model_word = pickle.load(file)
    with open('model_ngram.pkl', 'rb') as file:
        model_ngram = pickle.load(file)
    with open('model_chars.pkl', 'rb') as file:
        model_chars = pickle.load(file)
    return model_count, model_word, model_ngram, model_chars

# Function to predict sentiment
def predict_sentiment(models, text):
    model_count, model_word, model_ngram, model_chars = models
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Use each model to predict sentiment
    count_vectorizer = model_count.named_steps['vectorizer']
    word_vectorizer = model_word.named_steps['vectorizer']
    ngram_vectorizer = model_ngram.named_steps['vectorizer']
    chars_vectorizer = model_chars.named_steps['vectorizer']
    
    count_text = count_vectorizer.transform([processed_text])
    word_text = word_vectorizer.transform([processed_text])
    ngram_text = ngram_vectorizer.transform([processed_text])
    chars_text = chars_vectorizer.transform([processed_text])
    
    count_prediction = model_count.predict(count_text)
    word_prediction = model_word.predict(word_text)
    ngram_prediction = model_ngram.predict(ngram_text)
    chars_prediction = model_chars.predict(chars_text)
    
    return count_prediction[0], word_prediction[0], ngram_prediction[0], chars_prediction[0]

# Load models
models = load_models()

# Streamlit app
def main():
    st.title('Sentiment Analyzer')
    st.write('This app predicts the sentiment (positive or negative) of your input text.')

    # Input text box
    text = st.text_area('Enter text:')
    
    if st.button('Predict'):
        if text.strip() == '':
            st.warning('Please enter some text.')
        else:
            count_pred, word_pred, ngram_pred, chars_pred = predict_sentiment(models, text)
            
            st.subheader('Prediction Results:')
            st.write('Count Vectors Model Prediction:', 'Positive' if count_pred == 1 else 'Negative')
            st.write('TF-IDF Word Model Prediction:', 'Positive' if word_pred == 1 else 'Negative')
            st.write('TF-IDF Ngram Model Prediction:', 'Positive' if ngram_pred == 1 else 'Negative')
            st.write('TF-IDF Chars Model Prediction:', 'Positive' if chars_pred == 1 else 'Negative')

if __name__ == '__main__':
    main()
