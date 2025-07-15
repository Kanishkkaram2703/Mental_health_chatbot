from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__, static_folder='')
CORS(app)

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.index")
with open("answers.pkl", "rb") as f:
    answers = pickle.load(f)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_input = request.json['query']
    embedding = model.encode([user_input])
    D, I = index.search(np.array(embedding), k=1)
    answer = answers[I[0][0]]
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)