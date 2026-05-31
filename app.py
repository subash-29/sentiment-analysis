import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Flipkart Sentiment Analysis",
    page_icon="🛒",
    layout="wide"
)

# ------------------------------
# Download Stopwords
# ------------------------------
nltk.download('stopwords')

# ------------------------------
# Load Model & TFIDF
# ------------------------------
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# ------------------------------
# Stopwords
# ------------------------------
stop_words = set(stopwords.words('english'))

# ------------------------------
# Text Cleaning Function
# ------------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = text.strip()

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ------------------------------
# Layout
# ------------------------------
left_col, right_col = st.columns([2, 1])

# ------------------------------
# LEFT SIDE
# ------------------------------
with left_col:

    st.title("🛒 Flipkart Reviews Sentiment Analysis")

    st.write(
        "Enter a Flipkart review and predict whether "
        "it is Positive, Negative or Neutral."
    )

    review = st.text_area(
        "Enter Review",
        height=200
    )

    if st.button("Predict Sentiment"):

        if review.strip() == "":
            st.warning("Please enter a review.")

        else:

            cleaned_review = clean_text(review)

            vector = tfidf.transform([cleaned_review])

            prediction = model.predict(vector)

            sentiment = prediction[0]

            if sentiment.lower() == "positive":
                st.success("😊 Positive Review")

            elif sentiment.lower() == "negative":
                st.error("😠 Negative Review")

            else:
                st.info("😐 Neutral Review")

# ------------------------------
# RIGHT SIDE IMAGE
# ------------------------------
with right_col:

    st.subheader("Dataset Distribution")

    st.image(
        "sentimentdistribution.png",
        use_container_width=True
    )

    st.caption(
        "Distribution of Positive, Negative and Neutral reviews "
        "used during model training."
    )