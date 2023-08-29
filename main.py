import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the pre-trained model
loaded_model = pickle.load(open('DT_MODEL', 'rb'))

# Function to classify news
def fake_news_det(news):
    tfid_news = tfvect.transform([news])
    prediction = loaded_model.predict(tfid_news)
    probability = loaded_model.predict_proba(tfid_news)
    return prediction, probability

# Set page background image
page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Main title
st.title("NEWS BOT")

# Sidebar
st.sidebar.title("Options")
news_text = st.sidebar.text_area("Enter Text")

if st.sidebar.button("Classify"):
    if news_text:
        prediction, probability = fake_news_det(news_text)

        if prediction[0] == 0:
            st.error("This news is classified as FAKE.")
        else:
            st.success("This news is classified as TRUE.")

        st.subheader("Prediction Probability")
        st.write(f"Probability of being FAKE: {probability[0][0]:.2f}")
        st.write(f"Probability of being TRUE: {probability[0][1]:.2f}")

# Information and credits
st.sidebar.title("About")
st.sidebar.info("This app classifies news articles as FAKE or TRUE.")
st.sidebar.text("Dataset Source: Your Dataset Source")
st.sidebar.text("Model: Decision Tree Classifier")

# You can include additional information and references here.

# Background image
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Background image source: Your Image Source</p>", unsafe_allow_html=True)
