import faiss
import pickle
from sentence_transformers import SentenceTransformer
import subprocess


model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.index")
with open("answers.pkl", "rb") as f:
    answers = pickle.load(f)


def query_llama(user_input, top_answer):
    prompt = f"""
You are a kind and emotionally supportive best friend.

Respond directly to this concern in a gentle, calming, friendly way. Use soft emojis naturally within the sentences like 🤍😊🌿.

Speak as if you're texting a close friend. Do not repeat the question or mention any instructions.


Reply:
"""
    result = subprocess.run(["ollama", "run", "tinyllama"], input=prompt.encode(), stdout=subprocess.PIPE)
    return result.stdout.decode().strip()


def get_best_answer(user_input):
    emb = model.encode([user_input])
    D, I = index.search(emb, k=1)
    top_answer = answers[I[0][0]]
    print(" FAISS Match:", top_answer)
    final = query_llama(user_input, top_answer)
    print(" LLaMA Reply:", final)
    return final
