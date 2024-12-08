from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import numpy as np
from embedding import split_text_into_chunks, embed_text
from quering import find_most_relevant_chunk

app = Flask(__name__)

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure that these functions are correctly defined in embedding.py and quering.py
def load_and_embed_text():
    with open("C:/Users/abdul/Desktop/PROGRAMMING/SmartMedNotes/combined_books.txt", "r", encoding="utf-8") as file:
        combined_text = file.read()

    text_chunks = split_text_into_chunks(combined_text)
    embeddings = embed_text(text_chunks)

    return text_chunks, embeddings

# Load the text chunks and embeddings
text_chunks, embeddings = load_and_embed_text()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    result = find_most_relevant_chunk(query_text, text_chunks, embeddings, model)
    return jsonify({"answer": result})

if __name__ == '__main__':
    app.run(debug=True)
