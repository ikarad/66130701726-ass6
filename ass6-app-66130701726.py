import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pythainlp.corpus import thai_stopwords
import pickle
import re

def TextClaen(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

th_stop = list(thai_stopwords())
tfidf = TfidfVectorizer(use_idf=True, norm= 'l2', smooth_idf=True,stop_words=th_stop, preprocessor=TextClaen)


model = pickle.load(open('per_model-66130701726.sav', 'rb'))

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
