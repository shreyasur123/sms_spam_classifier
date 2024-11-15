import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Download NLTK resources only when needed
def preprocess_text(text):
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text) if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(text)

def main():
    st.set_page_config(page_title="SMS Spam Classifier")
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
                st.error("This message is classified as SPAM.")
                st.write(f"Probability: {probability:.2%}")
            else:
                st.success("This message is classified as NOT SPAM.")
                st.write(f"Probability: {(1 - probability):.2%}")
        else:
            st.warning("Please enter some text to classify.")

if __name__ == "__main__":
    main()
