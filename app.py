import streamlit as st
import mlflow
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import time
import pandas as pd
import os

if not os.path.exists("xnli_index.bin"):
    st.error("FAISS index missing. Run build_index.py first.")
    st.stop()

if not os.path.exists("xnli_metadata.pkl"):
    st.error("Metadata missing. Run build_index.py first.")
    st.stop()

index = faiss.read_index("xnli_index.bin")
with open("xnli_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

st.set_page_config(page_title="Multilingual RAG", page_icon="🌐", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 3rem; color: #1f77b4; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Multilingual NLP System</h1>', unsafe_allow_html=True)
st.info("2.5GB • 15 Languages • MLflow Tracking • Production RAG Pipeline")

@st.cache_resource
def init_mlflow():
    mlflow.set_experiment("Production_experiment")
    return True

init_mlflow()
st.sidebar.success("MLflow Active")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("intfloat/multilingual-e5-base")

embedder = load_embedder()
total_docs = index.ntotal
st.success(f"Loaded {total_docs:,} XNLI documents")

st.sidebar.header("MLflow Tracking")
track_queries = st.sidebar.checkbox("Log queries to MLflow", value=True)

if st.sidebar.button("Open MLflow UI"):
    st.sidebar.info("http://localhost:5000")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Multilingual Query")
    query = st.text_area(
        "Ask in ANY language:",
        placeholder="e.g., 'capital of India', 'भारत की राजधानी', '气候变化是什么'",
        height=100
    )
    k_results = st.slider("Top K Results", 3, 15, 5)

with col2:
    st.markdown("### Dataset Stats")
    st.metric("Documents", f"{total_docs:,}")
    st.metric("Languages", "15")
    st.metric("Size", "2.5GB")

def rag_search(query, k=5):
    if track_queries:
        with mlflow.start_run(nested=True) as active_run:
            mlflow.log_param("query", query[:200])
            mlflow.log_param("top_k", k)
            mlflow.log_param("dataset", "XNLI_2.5GB")
            run_id = active_run.info.run_id

    start_retrieval = time.time()
    q_emb = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
    scores, indices = index.search(q_emb, k)

    results = []
    for i, idx in enumerate(indices[0]):
        meta = metadata[idx]
        results.append({
            "score": float(scores[0][i]),
            "lang": meta["lang"],
            "premise": meta["premise"][:300],
            "id": meta["id"]
        })

    retrieval_time = (time.time() - start_retrieval) * 1000

    if track_queries:
        mlflow.log_metric("retrieval_time_ms", retrieval_time)
        mlflow.log_metric("top_score", results[0]["score"])
        mlflow.log_metric("avg_score", np.mean([r["score"] for r in results]))
        st.sidebar.success(f"Logged Run: {run_id[:8]}")

    return results

def generate_answer(query, results):
    context = "\n---\n".join([
        f"[{r['lang'].upper()}] Score: {r['score']:.3f}\n{r['premise']}" 
        for r in results[:5]
    ])
    
    answer = f"""MULTILINGUAL RAG ANSWER

QUERY: {query}

TOP RETRIEVED CONTEXT ({len(results)} documents):

{context}

SUMMARY: The most relevant XNLI documents across {len(set(r['lang'] for r in results))} languages 
have been retrieved. Top match score: {results[0]['score']:.3f} ({results[0]['lang'].upper()})."""
    
    return answer

if st.button("RUN RAG PIPELINE", type="primary", use_container_width=True) and query.strip():
    with st.spinner("Running RAG Pipeline..."):
        progress_bar = st.progress(0)
        progress_bar.progress(40)
        results = rag_search(query, k_results)
        progress_bar.progress(100)
        progress_bar.empty()
        
        answer = generate_answer(query, results)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top Language", results[0]["lang"].upper())
    with col2:
        st.metric("Top Score", f"{results[0]['score']:.3f}")
    with col2:
        st.metric("Retrieved Docs", len(results))

    st.markdown("### 📊 Retrieval Quality")
    scores_df = pd.DataFrame({
        "Rank": range(1, len(results) + 1),
        "Relevance Score": [r["score"] for r in results]
    })
    st.bar_chart(scores_df.set_index("Rank"))

    st.markdown("### 📚 Retrieved Documents")
    for i, result in enumerate(results):
        with st.expander(f"#{i+1} | {result['lang'].upper()} | {result['score']:.3f}"):
            st.markdown(f"**Preview:** {result['premise']}")
            st.caption(f"Doc ID: {result['id']}")

    st.markdown("### 🤖 Generated Answer")
    st.info(answer)

with st.expander("Quick Start"):   
    st.markdown("""
    **Terminal 1:**
    ```bash
    mlflow ui --port 5000
    ```
    ...
    """)
