import streamlit as st
import pickle
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = [] 
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

def main():
    st.title("SMS Spam Classifier")
    
    user_input = st.text_area("Enter the SMS text:")
    
    if st.button("Predict"):
        processed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([processed_input])
        prediction = model.predict(vectorized_input)[0]
        
        if prediction == 1:
            st.error("This message is classified as SPAM.")
        else:
            st.success("This message is classified as NOT SPAM.")

if __name__ == "__main__":
    main()
