# streamlit_app.py
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model("model.h5")
tokenizer = joblib.load("tokenizer.pkl")


# Function to predict sentiment
def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞"
    return sentiment

# Streamlit frontend
st.set_page_config(page_title="Movie Sentiment Analysis", page_icon="🎬", layout="centered")

st.title("🎬 MOVIE SENTIMENT ANALYSIS APPLICATION BY ADITYA")
st.write("Type a movie review below and see whether it's Positive or Negative!")

# User input
user_input = st.text_area("Enter a movie review:")

# Prediction button
if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        sentiment = predictive_system(user_input)
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review before clicking Predict.")
