from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import requests

# Inicializar MCP
mcp = FastMCP("Demo 🚀")

# Cargar modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cargar índice FAISS
index = faiss.read_index("docs.index")

# Cargar chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def search(query):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), 5)
    return [chunks[i] for i in I[0]]


@mcp.tool()
def search_docs(query: str):
    """Search in documentation"""
    return search(query)


if __name__ == "__main__":
    mcp.run()


@mcp.tool()
def scrape_web(url: str) -> str:
    """Download webpage content in markdown using Jina Reader"""

    jina_url = f"https://r.jina.ai/{url}"
    response = requests.get(jina_url)

    if response.status_code != 200:
        return f"Error fetching page: {response.status_code}"

    return response.text
