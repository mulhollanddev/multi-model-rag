"""
multi_model_rag.py
Exemplo minimal de RAG multi-modelo:
 - Embeddings: sentence-transformers (all-MiniLM-L6-v2)
 - Vector DB: FAISS (IndexFlatIP, cos sim via normalização)
 - Model A (rápido): Flan-T5-small (text2text) -> usado para pontuar / resposta rápida
 - Model B (final): Flan-T5-small (pode ser trocado por um modelo remoto via OpenAI)
 
Adaptações:
 - Para usar um modelo local maior (p.ex. Llama/Falcon) troque `pipeline` por seu runtime (HF, Ollama, etc).
 - Para usar OpenAI/GPT como Model B, preencha OPENAI_API_KEY e descomente a parte correspondente.
"""
import os
import glob
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# ---------- Utils: leitura e chunking ----------
def read_txts(folder: str) -> List[Tuple[str, str]]:
    """Retorna lista de (path, text) para .txt em folder."""
    paths = glob.glob(os.path.join(folder, "*.txt"))
    out = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            out.append((p, f.read()))
    return out

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Chunk por caracteres (simples e robusto). Ajuste chunk_size conforme precisa."""
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = min(i + chunk_size, L)
        chunks.append(text[i:end].strip())
        i += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]

# ---------- Embeddings e FAISS ----------
def build_embeddings_index(chunks: List[str], emb_model_name="all-MiniLM-L6-v2"):
    emb_model = SentenceTransformer(emb_model_name)
    embeddings = emb_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    # normalizar para usar Inner Product como cosine
    import faiss
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings, emb_model

def retrieve(query: str, index, emb_model, chunks: List[str], top_k: int = 5):
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)  # D: scores, I: indices
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({"idx": int(idx), "score": float(score), "chunk": chunks[int(idx)]})
    return results

# ---------- Model A: reranker / answer quick (fast local) ----------
def init_modelA():
    """Modelo rápido (local). Aqui usamos Flan-T5-small (text2text)."""
    pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=-1, max_length=200)
    return pipe

def score_with_modelA(modelA_pipe, query: str, chunk: str) -> float:
    """Pede ao modelA uma pontuação 1-5 de relevância. Retorna float."""
    prompt = (
        "Em uma escala de 1 a 5, onde 5 = muito relevante e 1 = irrelevante, "
        "avalie o quão relevante o seguinte trecho é para responder à pergunta.\n\n"
        f"Pergunta: {query}\nTrecho: {chunk}\nResposta:"
    )
    out = modelA_pipe(prompt, max_length=20)[0]["generated_text"].strip()
    # Tenta extrair número
    import re
    m = re.search(r"([1-5])", out)
    if m:
        return float(m.group(1))
    # fallback: usar similaridade textual como 0-5
    return 1.0

# ---------- Model B: gerador final (local ou remoto) ----------
def init_modelB_local():
    """Modelo final local (usamos também Flan-T5-small aqui por simplicidade)."""
    pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=-1, max_length=512)
    return pipe

def generate_answer_with_modelB_local(modelB_pipe, query: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "Use as informações do contexto abaixo para responder à pergunta. Se as informações forem incompletas, "
        "seja honesto e responda que você não tem informação suficiente.\n\n"
        f"Contexto:\n{context}\n\nPergunta: {query}\n\nResposta (seja conciso):"
    )
    out = modelB_pipe(prompt, max_length=400)[0]["generated_text"].strip()
    return out

# ---------- Híbrido: exemplo usando OpenAI ChatCompletion (opcional) ----------
def generate_answer_with_openai(api_key: str, query: str, context_chunks: List[str]):
    import openai
    openai.api_key = api_key
    context = "\n\n---\n\n".join(context_chunks)
    system = "Você é um assistente que usa apenas o contexto fornecido para responder. Se não houver informação suficiente, responda 'Não sei'."
    prompt = f"Contexto:\n{context}\n\nPergunta: {query}\nResposta:"
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # exemplo, ajuste conforme sua conta/perm
        messages=[{"role":"system","content":system}, {"role":"user","content":prompt}],
        max_tokens=400,
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()

# ---------- Orquestrador: combina tudo ----------
def multi_model_rag_query(query: str, index, emb_model, chunks: List[str],
                          modelA_pipe=None, modelB_pipe=None, openai_key=None,
                          top_k=5, rerank_with_modelA=False):
    # 1) retrieve
    retrieved = retrieve(query, index, emb_model, chunks, top_k=top_k)
    if not retrieved:
        return "Nenhum documento recuperado."

    # 2) opcional rerank por modelA
    if rerank_with_modelA and modelA_pipe is not None:
        for r in retrieved:
            r["modelA_score"] = score_with_modelA(modelA_pipe, query, r["chunk"])
        # combinar score FAISS (embedding) e modelA_score (ponderando)
        for r in retrieved:
            r["combined"] = 0.5 * r["score"] + 0.5 * (r["modelA_score"] / 5.0)  # normaliza modelA para 0-1
        retrieved = sorted(retrieved, key=lambda x: x["combined"], reverse=True)
    else:
        retrieved = sorted(retrieved, key=lambda x: x["score"], reverse=True)

    top_chunks = [r["chunk"] for r in retrieved[:top_k]]

    # 3) resposta rápida com Model A (opcional)
    quick_answer = None
    if modelA_pipe:
        # prompt simples pedindo resposta curta usando o primeiro chunk
        qprompt = (
            f"Use apenas o trecho abaixo para responder de forma curta à pergunta.\n\nTrecho: {top_chunks[0]}\n\nPergunta: {query}\nResposta:"
        )
        quick_answer = modelA_pipe(qprompt, max_length=150)[0]["generated_text"].strip()

    # 4) resposta final com Model B
    final_answer = None
    if modelB_pipe:
        final_answer = generate_answer_with_modelB_local(modelB_pipe, query, top_chunks)
    elif openai_key:
        final_answer = generate_answer_with_openai(openai_key, query, top_chunks)
    else:
        final_answer = quick_answer or "Nenhum gerador configurado."

    # 5) (opcional) comparar respostas (voto/simple confidence)
    return {
        "query": query,
        "retrieved": retrieved,
        "quick_answer": quick_answer,
        "final_answer": final_answer
    }

# ---------- Exemplo runnable ----------
def demo(folder="docs"):
    # 1) load docs (ou cria docs de exemplo se não houver)
    if os.path.isdir(folder) and glob.glob(os.path.join(folder, "*.txt")):
        docs = read_txts(folder)
        all_chunks = []
        for path, text in docs:
            ch = chunk_text(text, chunk_size=1000, overlap=200)
            all_chunks.extend(ch)
    else:
        print("Pasta docs/ não encontrada — usando textos de exemplo.")
        sample1 = "O Wikidata é um repositório de dados livres, estruturados, ligado à Wikimedia. Ele armazena itens e propriedades."
        sample2 = "RAG significa Retrieval Augmented Generation, uma técnica que combina recuperação de documentos com modelos de linguagem para respostas mais precisas."
        all_chunks = chunk_text(sample1, 800, 200) + chunk_text(sample2, 800, 200)

    # 2) index
    print("Construindo índice e embeddings...")
    index, embeddings, emb_model = build_embeddings_index(all_chunks)

    # 3) init models
    print("Iniciando Model A (rápido) e Model B (gerador)...")
    modelA = init_modelA()
    modelB = init_modelB_local()

    # 4) testar queries
    queries = [
        "O que é RAG e quando usá-lo?",
        "Para que serve o Wikidata?"
    ]
    for q in queries:
        print("\n=== QUERY:", q)
        out = multi_model_rag_query(q, index, emb_model, all_chunks, modelA_pipe=modelA, modelB_pipe=modelB, top_k=3, rerank_with_modelA=True)
        print("Quick answer (modelA):\n", out["quick_answer"])
        print("Final answer (modelB):\n", out["final_answer"])
        print("Top retrieved snippets (scores):")
        for r in out["retrieved"]:
            # mostra small preview do chunk
            preview = r["chunk"][:200].replace("\n", " ")
            print(f" - idx={r['idx']} faiss_score={r['score']:.3f} modelA_score={r.get('modelA_score',None)} preview='{preview}...'")

if __name__ == "__main__":
    demo()
