from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("clip-ViT-B-32")
    return _model


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/embed")
def embed():
    if "image" not in request.files:
        return jsonify({"error": "missing image"}), 400

    file = request.files["image"]
    if not file:
        return jsonify({"error": "invalid image"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "unable to read image"}), 400

    model = get_model()  # model loads ONLY when /embed is called
    embedding = model.encode(image)

    vec = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return jsonify({
        "model": "clip-ViT-B-32",
        "embedding": vec.tolist()
    })
