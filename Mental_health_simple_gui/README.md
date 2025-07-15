# Mental Health Chatbot – FAISS + LLaMA2 + Flask + Custom HTML UI

This project is a hybrid mental health chatbot using:

 FAISS + SentenceTransformers for fast answer retrieval.
 LLaMA2 via Ollama for enhancing chatbot replies.
 Flask app (`app.py`) for quick testing and GUI.
 Custom HTML/CSS frontend (`index.html`) for professional GUI integration.

## Project Structure

mental-health-chatbot-full/
├── app.py
├── chatbot_llama.py
├── faiss_index.index
├── answers.pkl
├── generate_faiss.py
├── mentalhealth.csv
├── index.html

## How It Works

 You embed all `Questions` from `mentalhealth.csv` using SentenceTransformers.
 Store embeddings using FAISS for quick similarity search.
 When a user types a question, FAISS returns the best match answer.
 Optionally, `chatbot_llama.py` enhances that answer using LLaMA2 model locally via Ollama.

## How To Run Locally

### Install Python Requirements

pip install faiss-cpu sentence-transformers streamlit flask flask-cors

### Install Ollama (for LLaMA2)

curl https://ollama.com/install.sh | sh
ollama pull llama2

### Generate FAISS Files

python generate_faiss.py

### Run Chatbot (Quick Test)
python app.py

### Using Custom Frontend (index.html)

Host `index.html` on any web server and link it to your Flask backend using `chatbot_llama.py` logic.

## Example Questions

 How do I handle anxiety?
 I am feeling stressed, what should I do?
 How can I manage overthinking?

## Important Notes

 Ollama must be running while using `chatbot_llama.py`.
 HTML UI are separate options.
 FAISS ensures quick retrieval for large datasets.

## ✍️ Author

 Kanishk Karam / SeKan
 https://github.com/Kanishkkaram2703

## ✅ License

MIT Licensed.