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

# Load the model from the pickle file
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_sentiments(text):
    """
    Lets analyze the text from the input 
    if the text is positive or negative.
    This uses Sentiment Analysis and docstring specifications.
    ---
    parameters:
      - name: text
        in: query
        required: true
    responses:
        200:
            description: The output text
    """   
    prediction = model.predict([text])
    print(prediction)
    return prediction

def main():
    st.title("Sentiment Analyzer")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Sentiment Analyzer ML App</h2>
    <h3 style="color:white;text-algin:center;">1 is postive 0 is negative</h3>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    text = st.text_input("Text", "Type Here")
    
    result = ""
    if st.button("Analyze"):
        result = predict_sentiments(text)
        st.success('The output is {}'.format(result))

if __name__ == '__main__':
    main()
