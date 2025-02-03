import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Set page config with icon and title
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Add custom CSS for animations
st.markdown("""
    <style>
    .title {
        animation: fadeIn 1.5s;
    }
    .result {
        animation: slideIn 1s;
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    @keyframes slideIn {
        0% {transform: translateX(-100%);}
        100% {transform: translateX(0);}
    }
    </style>
""", unsafe_allow_html=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Add styled title with icon
st.markdown('<h1 class="title">📧 Email/SMS Spam Classifier</h1>', unsafe_allow_html=True)

# Add a divider
st.markdown("---")

input_sms = st.text_area("Enter the message 💬")

if st.button('Predict 🔍'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display with animation
    if result == 1:
        st.markdown('<h2 class="result">⚠️ Spam</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="result">✅ Not Spam</h2>', unsafe_allow_html=True)