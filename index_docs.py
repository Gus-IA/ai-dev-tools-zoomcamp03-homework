import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Carpeta con markdown
repo_path = "repo_docs"
chunks = []

# Leer todos los archivos markdown
for root, dirs, files in os.walk(repo_path):
    for file in files:
        if file.endswith(".md"):
            with open(os.path.join(root, file), encoding="utf-8") as f:
                text = f.read()
                words = text.split()
                for i in range(0, len(words), 500):
                    chunks.append(" ".join(words[i : i + 500]))

# Guardar chunks
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

# Crear embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True)

# Asegurar array 2D
if embeddings.ndim == 1:
    embeddings = embeddings.reshape(1, -1)

print("Shape de embeddings:", embeddings.shape)

# Crear índice FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Guardar índice
faiss.write_index(index, "docs.index")
print("chunks.pkl y docs.index generados correctamente")
