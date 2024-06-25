import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib  # For older scikit-learn versions
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import Word
import nltk
import emoji

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

# Function to preprocess text
def text_preprocessing(text):
    text = emoji.replace_emoji(str(text), '') if isinstance(text, str) else ''
    text = " ".join(text.lower() for text in str(text).split())
    text = text.replace('[^\w\s]', '')
    text = text.replace('\d', '')
    stop_words = set(stopwords.words('english'))
    text = " ".join(word for word in text.split() if word not in stop_words)
    text = " ".join(Word(word).lemmatize() for word in text.split())
    return text

# Load models and vectorizers
def load_models():
    # Load CountVectorizer and LogisticRegression model
    with open('model_count.pkl', 'rb') as file:
        model_count = joblib.load(file)
    with open('vectorizer_count.pkl', 'rb') as file:
        vectorizer_count = joblib.load(file)

    # Load TF-IDF Word Vectorizer and LogisticRegression model
    with open('model_word.pkl', 'rb') as file:
        model_word = joblib.load(file)
    with open('vectorizer_word.pkl', 'rb') as file:
        vectorizer_word = joblib.load(file)

    # Load TF-IDF Ngram Vectorizer and LogisticRegression model
    with open('model_ngram.pkl', 'rb') as file:
        model_ngram = joblib.load(file)
    with open('vectorizer_ngram.pkl', 'rb') as file:
        vectorizer_ngram = joblib.load(file)

    # Load TF-IDF Character Vectorizer and LogisticRegression model
    with open('model_chars.pkl', 'rb') as file:
        model_chars = joblib.load(file)
    with open('vectorizer_chars.pkl', 'rb') as file:
        vectorizer_chars = joblib.load(file)

    return {
        'count': {'model': model_count, 'vectorizer': vectorizer_count},
        'word': {'model': model_word, 'vectorizer': vectorizer_word},
        'ngram': {'model': model_ngram, 'vectorizer': vectorizer_ngram},
        'chars': {'model': model_chars, 'vectorizer': vectorizer_chars}
    }

# Predict sentiment using the selected model and vectorizer
def predict_sentiment(model, vectorizer, text):
    preprocessed_text = text_preprocessing(text)
    transformed_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(transformed_text)[0]
    return prediction

# Streamlit UI
def main():
    st.title("Sentiment Analysis App")
    st.write("This app predicts the sentiment (positive/negative) of input text.")

    text = st.text_area("Enter your text here:", "Type here")

    models = load_models()
    options = ['count', 'word', 'ngram', 'chars']
    selected_model = st.selectbox("Select a model:", options)

    if st.button("Analyze"):
        result = predict_sentiment(models[selected_model]['model'], models[selected_model]['vectorizer'], text)
        if result == 1:
            st.success('The comment is Positive')
        else:
            st.error('The comment is Negative')

if __name__ == '__main__':
    main()
