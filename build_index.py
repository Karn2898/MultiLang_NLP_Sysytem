import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


embedder = SentenceTransformer("intfloat/multilingual-e5-base")


texts = [
    "India is a country in South Asia.",
    "The capital of India is New Delhi.",
    "Climate change affects global weather patterns."
]


embeddings = embedder.encode(
    texts,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")


dim = embeddings.shape[1]  
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "xnli_index.bin")

metadata = [
    {"id": i, "premise": texts[i], "lang": "en"}
    for i in range(len(texts))
]

with open("xnli_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(" FAISS index rebuilt with E5 embeddings")
