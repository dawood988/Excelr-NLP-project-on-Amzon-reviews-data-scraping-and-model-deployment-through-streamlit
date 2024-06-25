from warnings import filterwarnings
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
import emoji
import pickle

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

# Suppress warnings
filterwarnings("ignore")

# Load data
df_reviews = pd.read_csv("reviews.csv")
df = pd.DataFrame(df_reviews['Description'])

# Function to preprocess text data
def text_preprocessing(dataframe, dependent_var):
    dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
    dataframe[dependent_var] = dataframe[dependent_var].str.replace('[^\w\s]', '', regex=True)
    dataframe[dependent_var] = dataframe[dependent_var].str.replace('\d', '', regex=True)
    sw = stopwords.words('english')
    dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    temp_df = pd.Series(' '.join(dataframe[dependent_var]).split()).value_counts()
    drops = temp_df[temp_df <= 1]
    dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))
    dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return dataframe

# Preprocess text
df = text_preprocessing(df, "Description")

# Function to create sentiment labels
def create_label(dataframe, dependent_var, independent_var):
    sia = SentimentIntensityAnalyzer()
    dataframe[independent_var] = dataframe[dependent_var].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
    dataframe[independent_var] = LabelEncoder().fit_transform(dataframe[independent_var])
    X = dataframe[dependent_var]
    y = dataframe[independent_var]
    return X, y

# Create sentiment labels
X, y = create_label(df, "Description", "sentiment_label")

# Split dataset
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=1)

# Function to create features using Count Vectorizer
def create_features_count(train_x, test_x):
    vectorizer = CountVectorizer()
    x_train_count_vectorizer = vectorizer.fit_transform(train_x)
    x_test_count_vectorizer = vectorizer.transform(test_x)
    return x_train_count_vectorizer, x_test_count_vectorizer, vectorizer

# Function to create features using TF-IDF Vectorizer (word level)
def create_features_TFIDF_word(train_x, test_x):
    tf_idf_word_vectorizer = TfidfVectorizer()
    x_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(train_x)
    x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)
    return x_train_tf_idf_word, x_test_tf_idf_word, tf_idf_word_vectorizer

# Function to create features using TF-IDF Vectorizer (ngram level)
def create_features_TFIDF_ngram(train_x, test_x):
    tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
    x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(train_x)
    x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)
    return x_train_tf_idf_ngram, x_test_tf_idf_ngram, tf_idf_ngram_vectorizer

# Function to create features using TF-IDF Vectorizer (character level)
def create_features_TFIDF_chars(train_x, test_x):
    tf_idf_chars_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    x_train_tf_idf_chars = tf_idf_chars_vectorizer.fit_transform(train_x)
    x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)
    return x_train_tf_idf_chars, x_test_tf_idf_chars, tf_idf_chars_vectorizer

# Train logistic regression models using different vectorization techniques
def crate_model_logistic(train_x, train_y, test_x, test_y):
    # Count Vectorizer
    x_train_count_vectorizer, x_test_count_vectorizer, _ = create_features_count(train_x, test_x)
    loj_count = LogisticRegression(solver='lbfgs', max_iter=1000)
    loj_model_count = loj_count.fit(x_train_count_vectorizer, train_y)
    accuracy_count = cross_val_score(loj_model_count, x_test_count_vectorizer, test_y, cv=10).mean()
    print("Accuracy - Count Vectors: %.3f" % accuracy_count)

    # TF-IDF Word Vectorizer
    x_train_tf_idf_word, x_test_tf_idf_word, _ = create_features_TFIDF_word(train_x, test_x)
    loj_word = LogisticRegression(solver='lbfgs', max_iter=1000)
    loj_model_word = loj_word.fit(x_train_tf_idf_word, train_y)
    accuracy_word = cross_val_score(loj_model_word, x_test_tf_idf_word, test_y, cv=10).mean()
    print("Accuracy - TF-IDF Word: %.3f" % accuracy_word)

    # TF-IDF Ngram Vectorizer
    x_train_tf_idf_ngram, x_test_tf_idf_ngram, _ = create_features_TFIDF_ngram(train_x, test_x)
    loj_ngram = LogisticRegression(solver='lbfgs', max_iter=1000)
    loj_model_ngram = loj_ngram.fit(x_train_tf_idf_ngram, train_y)
    accuracy_ngram = cross_val_score(loj_model_ngram, x_test_tf_idf_ngram, test_y, cv=10).mean()
    print("Accuracy - TF-IDF Ngram: %.3f" % accuracy_ngram)

    # TF-IDF Chars Vectorizer
    x_train_tf_idf_chars, x_test_tf_idf_chars, _ = create_features_TFIDF_chars(train_x, test_x)
    loj_chars = LogisticRegression(solver='lbfgs', max_iter=1000)
    loj_model_chars = loj_chars.fit(x_train_tf_idf_chars, train_y)
    accuracy_chars = cross_val_score(loj_model_chars, x_test_tf_idf_chars, test_y, cv=10).mean()
    print("Accuracy - TF-IDF Chars: %.3f" % accuracy_chars)

    return loj_model_count, loj_model_word, loj_model_ngram, loj_model_chars

# Train models
loj_model_count, loj_model_word, loj_model_ngram, loj_model_chars = crate_model_logistic(train_x, train_y, test_x, test_y)

# Function to predict sentiment using Count Vectorizer model
def predict_count(model, new_comment):
    new_comment = pd.Series(new_comment)
    new_comment = count_vectorizer.transform(new_comment)
    result = model.predict(new_comment)
    if result == 1:
        return "Positive"
    else:
        return "Negative"

# Streamlit app
def main():
    # Title and introduction
    st.title("Sentiment Analyzer using Logistic Regression")
    st.markdown("""
        This is a Streamlit web app to predict sentiment (Positive or Negative) based on user input using Logistic Regression models trained on different vectorization techniques (Count Vectorizer, TF-IDF Word, TF-IDF Ngram, TF-IDF Chars).
        """)

    # User input for prediction
    user_input = st.text_area("Enter your text here:", "Type here...")

    if st.button("Predict"):
        # Predict sentiment using Count Vectorizer model
        prediction_count = predict_count(loj_model_count, user_input)
        st.write(f"Prediction using Count Vectors: {prediction_count}")

        # Additional predictions can be added for other models (Word, Ngram, Chars) if desired

    # Display acknowledgements or additional information
    st.markdown("""
        * Dataset source: [Kaggle](https://www.kaggle.com)
        * Built with Streamlit, NLTK, scikit-learn, and other libraries.
        """)

if __name__ == "__main__":
    main()
