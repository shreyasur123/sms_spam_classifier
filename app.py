import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def preprocess_text(text):
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text) if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(text)

def main():
    st.set_page_config(page_title="SMS Spam Classifier", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Roboto', sans-serif;
    }
    h1 {
        color: #333;
        font-weight: bold;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .stTextArea, .stButton > button {
        border-radius: 8px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-size: 1rem;
        padding: 1rem;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .result {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("SMS Spam Classifier")

    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    user_input = st.text_area("Enter the SMS text:", height=200)

    if st.button("Predict"):
        if user_input.strip():
            processed_input = preprocess_text(user_input)
            vectorized_input = vectorizer.transform([processed_input])
            prediction = model.predict(vectorized_input)[0]
            probability = model.predict_proba(vectorized_input)[0][prediction]

            if prediction == 1:
                st.markdown('<div class="result" style="color:#dc3545;">This message is classified as SPAM.</div>', unsafe_allow_html=True)
                st.write(f"Probability: {probability:.2%}")
            else:
                st.markdown('<div class="result" style="color:#28a745;">This message is classified as NOT SPAM.</div>', unsafe_allow_html=True)
                st.write(f"Probability: {probability:.2%}")
        else:
            st.warning("Please enter some text to classify.")

if __name__ == "__main__":
    main()
