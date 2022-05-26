import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from notebooks.fake_news_utils import *


st.header("Fake news detection")


st.subheader("This is a demo of the fake news detection pipeline")

st.write('''A few fake news sources:
         <ul>
         <li><a href="https://deadlyclear.com"> Deadly clear </a></li>
         <li><a href="http://www.jimstoneindia.com"> Questionable mental health </a></li>
         <li><a href="https://realnewsrightnow.com"> Real News Right Now </a></li>
         <li><a href="https://boards.4chan.org/pol/catalog"> 4chan - <strong>NOT RECOMMENDED</strong> </a></li>
         </ul>''', unsafe_allow_html=True)

text_to_detect = st.text_area("Enter text to classify as fake or not", "")

torch_checkpoint = "datasets/bert_clf-second.pth"



if text_to_detect:
    # Load from file
    bert_clf = torch.load(torch_checkpoint)
    bert_clf = bert_clf.to('cpu')

    # Preprocess text
    tokens_tensor, masks_tensor = preprocess_text_inference(text_to_detect)

    logits = bert_clf(tokens_tensor, masks_tensor)
    loss_func = nn.BCELoss()

    numpy_logits = logits.cpu().detach().numpy()

    bert_predicted = list(numpy_logits[:, 0] > 0.5)
    all_logits = list(numpy_logits[:, 0])


    st.subheader("Prediction")
    st.write('bert_predicted:', bert_predicted[0])
    st.write('all logits:', all_logits)
    st.write('Fake' if all_logits[0] < 0.5 else 'Real')


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
