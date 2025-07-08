import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load FAISS index and answers
index = faiss.read_index("faiss_index.index")
with open("answers.pkl", "rb") as f:
    answers = pickle.load(f)

# Load same embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("Mental Health Chatbot")

user_input = st.text_input("Your question:", placeholder="How do I deal with anxiety?")
if st.button("Send") and user_input:
    with st.spinner("Bot is thinking..."):
        # Embed the question
        query_embedding = model.encode([user_input])
        # Search in FAISS index
        D, I = index.search(np.array(query_embedding), k=1)
        response = answers[I[0][0]]
        st.markdown(f"**Bot:** {response}")

# Optional: Exit button
if st.button("Exit Chatbot"):
    st.warning("App is exiting...")
    import os
    os._exit(0)
