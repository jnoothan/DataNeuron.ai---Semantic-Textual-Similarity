from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer("all-mpnet-base-v2")

@app.route("/similarity", methods=["GET", "POST"])
def similarity():
    if request.method == "POST":
        # Handle JSON POST (e.g., from Postman)
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

        # Handle HTML form POST
        else:
            text1 = request.form.get("text1")
            text2 = request.form.get("text2")
            if not text1 or not text2:
                return render_template("similarity.html", similarity_score=None)
            emb1 = model.encode(text1, convert_to_tensor=True)
            emb2 = model.encode(text2, convert_to_tensor=True)
            score = util.cos_sim(emb1, emb2).item()
            return render_template("similarity.html", similarity_score=score)

    # GET request renders the form
    return render_template("similarity.html")
