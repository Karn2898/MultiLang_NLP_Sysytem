import streamlit as st
import mlflow
import numpy as np
import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import time
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="🌐 XNLI Multilingual RAG + MLflow",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
<style>
.main-header { font-size: 3rem; color: #1f77b4; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">XNLI Multilingual RAG</h1>', unsafe_allow_html=True)
st.info("2.5GB • 15 Languages • MLflow Tracking • Production RAG Pipeline")

@st.cache_resource
def init_mlflow():
    mlflow.set_experiment("Production_experiment")
    return True

init_mlflow()
st.sidebar.success("MLflow: XNLI_RAG_Production")

@st.cache_resource
def load_rag_pipeline():
    embedder = SentenceTransformer("intfloat/multilingual-e5-base")
    generator = pipeline(
        "text2text-generation",
        model="google/mt5-base",
        device=0 if torch.cuda.is_available() else -1
    )
    index = faiss.read_index("xnli_index.bin")
    with open("xnli_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return embedder, generator, index, metadata

embedder, generator, index, metadata = load_rag_pipeline()
total_docs = index.ntotal

st.sidebar.header("MLflow Tracking")
track_queries = st.sidebar.checkbox("Log queries to MLflow", value=True)
mlflow_link = st.sidebar.empty()

if st.sidebar.button("Open MLflow UI"):
    mlflow_link.info("http://localhost:5000")

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
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
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
    context = "\n".join(
        f"[{r['lang'].upper()}] {r['premise']} (score: {r['score']:.3f})"
        for r in results[:4]
    )

    prompt = f"""XNLI Multilingual Knowledge Base (15 languages):
{context}

Query: {query}

Answer:"""

    generation_kwargs = {
        "max_new_tokens": 150,
        "do_sample": True,
        "temperature": 0.7,
        "pad_token_id": generator.tokenizer.eos_token_id
    }

    output = generator(prompt, **generation_kwargs)[0]["generated_text"]
    return output[len(prompt):].strip()

if st.button("RUN RAG PIPELINE", type="primary", use_container_width=True) and query:
    with st.spinner("Running RAG Pipeline..."):
        results = rag_search(query, k_results)
        answer = generate_answer(query, results)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top Language", results[0]["lang"].upper())
    with col2:
        st.metric("Top Score", f"{results[0]['score']:.3f}")
    with col3:
        st.metric("Retrieved Docs", len(results))

    st.markdown("### Retrieval Quality")
    scores_df = pd.DataFrame({
        "Rank": range(1, len(results) + 1),
        "Relevance Score": [r["score"] for r in results]
    })
    st.bar_chart(scores_df.set_index("Rank"))

    st.markdown("### Retrieved Documents")
    for i, result in enumerate(results):
        with st.expander(f"{i+1} | {result['lang'].upper()} | {result['score']:.3f}"):
            st.markdown(result["premise"])
            st.caption(f"Doc ID: {result['id']}")

    st.markdown("### Generated Answer")
    st.info(answer)

with st.expander("Quick Start", expanded=False):
    st.markdown("""
Terminal 1:
mlflow ui --port 5000

Terminal 2:
streamlit run app.py

Queries:
capital of India
भारत की अर्थव्यवस्था
气候变化的影响

MLflow:
http://localhost:5000
""")
