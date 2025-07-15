import faiss
import pickle
from sentence_transformers import SentenceTransformer
import subprocess

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.index")

with open("answers.pkl", "rb") as f:
    answers = pickle.load(f)

def query_llama(prompt):
    command = ['ollama', 'run', 'llama2']
    full_prompt = f"Answer as a kind mental health advisor:\n{prompt}"
    result = subprocess.run(command, input=full_prompt.encode(), stdout=subprocess.PIPE)
    return result.stdout.decode()

def get_best_answer(user_input):
    emb = model.encode([user_input])
    D, I = index.search(emb, k=1)
    top_answer = answers[I[0][0]]
    full_prompt = f"User: {user_input}\nAnswer: {top_answer}\nNow enhance this answer."
    return query_llama(full_prompt)
