import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

import numpy as np
import re

max_len = 100

model = tf.keras.models.load_model('sentiment_model.keras')
tokenizer = load('tokenizer.joblib')
label_encoder = load('label_encoder.joblib')

def clean_text(text):
    return re.sub(r'[^a-z\s]','',text.lower())

def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen = max_len, padding = 'post')
    probs = model.predict(padded, verbose = 0)[0]
    predicted_class = np.argmax(probs)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label, probs

st.set_page_config(page_title = "Sentiment Analysis tool")
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet for analysis")

text_input = st.text_area("Your Input Here!")

if st.button("Execute"):
    if text_input.strip():
        sentiment, probs = predict_sentiment(text_input)
        st.success(f"Predicted Sentiment: {sentiment}")
        st.write("Confidence Scores:")
        for label, score in zip(label_encoder.classes_, probs):
            st.write(f"{label}: {score:.2f}")
    else:
        st.warning("Please enter some text!")














