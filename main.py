import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import streamlit as st

# Load the imdb dataset word index
word_index = imdb.get_word_index()
reverse__word_index = {value:key for key,value in word_index.items()}

#Load the pre-trained model
model = load_model('Simple_RNN.h5')

def decode_review(encoded_review):
    return ' '.join([reverse__word_index.get(i-3,'?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# prediction function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment = 'Postive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]

# streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as postive or negative')

# user input
user_input = st.text_area('Movie Review')
if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)
    # Make prediction
    prediction = model.predict(preprocess_input)
    sentiment ="Postive" if prediction[0][0]>0.5 else "Negative"

    #Display result
    st.write(f"Sentiment : {sentiment}")
    st.write(f"Prediction score : {prediction[0][0]}")
else:
    st.write("Please enter a movie review")