import pandas as pd
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
model = SentenceTransformer('all-mpnet-base-v2')

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    if not text1 or not text2:
        return jsonify({'error': 'Both "text1" and "text2" are required'}), 400
    
    # Generate embeddings
    embeddings = model.encode([text1, text2])
    
    # Compute cosine similarity
    raw_score = float(util.cos_sim(embeddings[0], embeddings[1]))
    
    print("Cosine score:", raw_score)

    # Return in expected response format
    return jsonify({'similarity score': raw_score})

if __name__ == '__main__':
    app.run(debug=True)
