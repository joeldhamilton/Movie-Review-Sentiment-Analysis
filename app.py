import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🎬", layout="centered")

# Title and description
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("""
Enter a movie review below, and this app will predict whether the sentiment is **positive** or **negative** using a custom-trained BERT model from my TripleTen project.
""")

# Preprocessing function from your notebook
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# BERT embedding function from your notebook
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def BERT_text_to_embeddings(texts, max_length=512, disable_progress_bar=False):
    tokenizer, model = load_bert_model()
    model.eval()
    embeddings = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=max_length, 
                          truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(embedding[0])
    
    return np.array(embeddings)

# Load the trained Logistic Regression model
@st.cache_resource
def load_logistic_model():
    return joblib.load('model_9_logistic_regression.joblib')

model_9 = load_logistic_model()

# Text input for user
user_input = st.text_area("Enter your movie review:", 
                         placeholder="e.g., This movie was thrilling and kept me on the edge of my seat!")

# Predict sentiment when the user clicks the button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Preprocess the input
        processed_text = preprocess_text(user_input)
        
        # Generate BERT embeddings
        embeddings = BERT_text_to_embeddings([processed_text])
        
        # Predict sentiment
        pred_prob = model_9.predict_proba(embeddings)[:, 1][0]
        label = "POSITIVE" if pred_prob >= 0.5 else "NEGATIVE"
        
        # Display result
        st.write("### Result")
        st.write(f"**Sentiment**: {label} (Confidence: {pred_prob:.2%})")
        
        # Add a friendly message
        if label == "POSITIVE":
            st.success("This review sounds like a glowing recommendation!")
        else:
            st.error("This review seems to express some disappointment.")
    else:
        st.warning("Please enter a review to analyze.")

# Footer
st.markdown("""
---
**Built by Joel Hamilton** | [GitHub](https://github.com/joeldhamilton) | [LinkedIn](https://www.linkedin.com/in/joel-hamilton) | [Portfolio](https://joeldhamilton.github.io)
""")