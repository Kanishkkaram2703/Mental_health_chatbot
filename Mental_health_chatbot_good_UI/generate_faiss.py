import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import pickle

df = pd.read_csv("mentalhealth.csv")
questions = df["Questions"].tolist()
answers = df["Answers"].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(questions)

index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(embeddings)

faiss.write_index(index, "faiss_index.index")
with open("answers.pkl", "wb") as f:
    pickle.dump(answers, f)

print("FAISS index and answers saved!")
