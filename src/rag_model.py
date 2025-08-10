from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        chunks.append(text[i:end].strip())
        i += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]

def build_index(chunks):
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = emb_model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, emb_model

def retrieve(query, index, emb_model, chunks, top_k=3):
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    return [chunks[idx] for idx in I[0] if idx != -1]

def init_model(name, max_len):
    return pipeline("text2text-generation", model=name, device=-1, max_length=max_len)
