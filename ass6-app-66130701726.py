import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

model,tfidf = pickle.load(open('per_model-66130701726.sav', 'rb'))



st.title('Text Classification LogisticRegression App')

# Text input for user to enter text
text = st.text_area('Enter your text here:')

if st.button('Classify'):
    # Transform the input text using the TfidfVectorizer
    text_tfidf = tfidf.transform([text])

    # Make prediction using the loaded model
    prediction = model.predict(text_tfidf)

    # Display the prediction
    if prediction[0] == 'neg':
        st.write('Negative')
    else:
        st.write('Positive')
