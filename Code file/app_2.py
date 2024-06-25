# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:50:58 2024

@author: Dawood M D
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from textblob import Word
import nltk
from nltk.corpus import stopwords
import emoji

nltk.download("stopwords")
nltk.download("wordnet")

# Load the count vectorizer and model from the pickle files
with open("model_word.pkl", "rb") as file_1:
    count_vectorizer = pickle.load(file_1)

with open("model_count.pkl", "rb") as file_2:
    model = pickle.load(file_2)

def text_preprocessing(text):
    """
    Perform the same preprocessing steps on the input text as was done on the training data.
    """
    sw = stopwords.words('english')

    # Replace emojis with an empty string
    text = emoji.replace_emoji(str(text), '')

    # Normalize case folding - uppercase to lowercase
    text = " ".join(x.lower() for x in text.split())

    # Remove punctuation
    text = ''.join([c for c in text if c not in ('!', '.', ':', ',', ';', '?', '"', "'")])

    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])

    # Remove stopwords
    text = " ".join(x for x in text.split() if x not in sw)

    # Lemmatize
    text = " ".join([Word(word).lemmatize() for word in text.split()])

    return text

def predict_sentiments(text):
    """
    Lets analyze the text from the input 
    if the text is positive or negative.
    """
    preprocessed_text = text_preprocessing(text)
    transformed_text = count_vectorizer.transform([preprocessed_text])
    prediction = model.predict(transformed_text)
    return prediction

def main():
    st.title("Sentiment Analyzer")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Sentiment Analyzer ML App</h2>
    <h3 style="color:white;text-align:center;">1 is positive, 0 is negative</h3>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    text = st.text_input("Text", "Type Here")
    
    result = ""
    if st.button("Analyze"):
        result = predict_sentiments(text)
        if result == 1:
            st.success('The comment is Positive')
        else:
            st.success('The comment is Negative')

if __name__ == '__main__':
    main()
