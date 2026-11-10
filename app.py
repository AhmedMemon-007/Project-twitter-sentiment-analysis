# app.py
# Run on your computer with:  streamlit run app.py

import pickle

import streamlit as st

from cleaning import clean_text


# Load the model only one time, so the app stays fast
@st.cache_resource
def load_model():
    with open("sentiment.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

# Colors for each label
colors = {
    "Positive": "green",
    "Negative": "red",
    "Neutral": "gray",
    "Irrelevant": "orange",
}

st.title("Twitter Sentiment Analysis")
st.write("Type a tweet. The model says if it is Positive, Negative, Neutral or Irrelevant.")

tweet = st.text_area("Your tweet:", height=120)

if st.button("Check sentiment"):
    if tweet.strip() == "":
        st.warning("Please type a tweet first.")
    else:
        cleaned = clean_text(tweet)           # clean the same way as training
        result = model.predict([cleaned])[0]  # the model is a full pipeline (TF-IDF + Linear SVC)

        st.markdown(f"### Sentiment: :{colors.get(result, 'blue')}[{result}]")

        with st.expander("See the cleaned text"):
            st.write(cleaned if cleaned else "(nothing left after cleaning)")

st.caption("Model: TF-IDF + Linear SVC, trained on the Kaggle Twitter Entity Sentiment dataset.")