from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

try:
    model = SentenceTransformer("all-mpnet-base-v2")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/', methods=["GET", "POST"])
def similarity():
    if model is None:
        return jsonify({"error": "Model not available."}), 500

    if request.method == "POST":
        # JSON request
        if request.is_json:
            data = request.get_json()
            text1 = data.get("text1")
            text2 = data.get("text2")
            if not text1 or not text2:
                return jsonify({"error": "Both 'text1' and 'text2' are required."}), 400
            emb1 = model.encode(text1, convert_to_tensor=True)
            emb2 = model.encode(text2, convert_to_tensor=True)
            score = util.cos_sim(emb1, emb2).item()
            return jsonify({"similarity_score": round(score, 4)})

        # HTML form
        text1 = request.form.get("text1")
        text2 = request.form.get("text2")
        if not text1 or not text2:
            return render_template("UI.html", similarity_score=None)
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()
        return render_template("UI.html", similarity_score=round(score, 4))

    # GET request
    return render_template("UI.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
