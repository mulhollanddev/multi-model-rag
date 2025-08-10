import streamlit as st
from src.rag_model import chunk_text, build_index, retrieve, init_model

def run_app():
    st.set_page_config(page_title="Multi-Model RAG", page_icon="📚")
    st.title("📚 Multi-Model RAG (Local)")

    uploaded_files = st.sidebar.file_uploader("Envie arquivos .txt", type=["txt"], accept_multiple_files=True)

    all_chunks = []
    if uploaded_files:
        for file in uploaded_files:
            text = file.read().decode("utf-8")
            all_chunks.extend(chunk_text(text))
        
        if st.sidebar.button("🔍 Criar índice"):
            with st.spinner("Criando índice..."):
                index, emb_model = build_index(all_chunks)
                modelA = init_model("google/flan-t5-small", 150)
                modelB = init_model("google/flan-t5-small", 400)

            st.session_state.update({
                "index": index,
                "emb_model": emb_model,
                "chunks": all_chunks,
                "modelA": modelA,
                "modelB": modelB
            })
            st.sidebar.success("Índice criado!")

    query = st.text_input("Digite sua pergunta:")

    if not uploaded_files:
        st.info("⚠️ Para fazer perguntas, envie ao menos um arquivo .txt e crie o índice.")

    elif "index" in st.session_state and query:
        top_chunks = retrieve(query, st.session_state["index"], st.session_state["emb_model"], st.session_state["chunks"])
        qa = st.session_state["modelA"](f"Pergunta: {query}\nTexto: {top_chunks[0]}")[0]["generated_text"]
        fa = st.session_state["modelB"](f"Pergunta: {query}\nContexto: {' '.join(top_chunks)}")[0]["generated_text"]

        st.subheader("Resposta Rápida")
        st.write(qa)
        st.subheader("Resposta Final")
        st.write(fa)
