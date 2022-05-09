import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


st.header("Fake news detection")
st.write("This is a demo of the fake news detection pipeline")

text_to_detect = st.text_area("Enter text to classify as fake or not", "")

pkl_filename = "pickle_model.pkl"

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


if text_to_detect:
    ps = PorterStemmer()

    review = re.sub('[^a-zA-Z]', ' ', text_to_detect)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)

    text_to_detect = [review]
    pkl_filename = "tfidf_v.pkl"
    with open(pkl_filename, 'rb') as file:
        tfidf_v = pickle.load(file)

    X = tfidf_v.transform(text_to_detect).toarray()
    st.subheader("Prediction")
    st.write('Fake' if pickle_model.predict(X) < 0.5 else 'Real')

