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

# Page config
st.set_page_config(
    page_title="🌐 MLQA Multilingual RAG", 
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌐 MLQA Multilingual Semantic Search")
st.markdown("**5K Real QA pairs • 7 Languages • MLflow Tracking • HuggingFace**")

# Custom CSS
st.markdown("""
<style>
    .metric-card { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; }
</style>
""", unsafe_allow_html=True)

# MLflow setup
mlflow.set_experiment("MLQA_RAG_Production")

@st.cache_resource
def load_mlqa_rag():
    """Load MLQA RAG pipeline with MLflow tracking"""
    # Models
    embedder = SentenceTransformer('intfloat/multilingual-e5-small')
    generator = pipeline("text2text-generation", model="google/mt5-small")
    
    # Load index & metadata
    index = faiss.read_index("mlqa_index.bin")
    with open("mlqa_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    return {
        "embedder": embedder,
        "generator": generator,
        "index": index,
        "metadata": metadata,
        "total_docs": index.ntotal
    }

# Load RAG
rag = load_mlqa_rag()
st.sidebar.success(f"✅ Loaded {rag['total_docs']} MLQA documents")

# Sidebar: MLflow Controls
st.sidebar.header("🔧 MLflow Tracking")
track_metrics = st.sidebar.checkbox("Track this query in MLflow", value=True)

if st.sidebar.button("📊 View MLflow Experiments"):
    st.sidebar.info("👉 Open http://localhost:5000 in new tab")

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_area(
        "💬 Search in ANY language:",
        placeholder="e.g., 'capital of India', 'भारत की राजधानी', 'AIとは'",
        height=80
    )
    k_results = st.slider("Top K results", 3, 10, 5)

with col2:
    st.metric("Documents", rag['total_docs'])
    st.metric("Languages", len(set(m['lang'] for m in rag['metadata'])))
    
    if st.button("🚀 SEARCH & TRACK", type="primary", use_container_width=True):
        pass

# Search function
def search_mlqa(query, k=5):
    """Semantic search with MLflow logging"""
    if track_metrics:
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param("query", query[:100])
            mlflow.log_param("top_k", k)
        
        start_time = time.time()
    
    # Embed query
    q_emb = rag['embedder'].encode([query], normalize_embeddings=True).astype('float32')
    
    # FAISS search
    scores, indices = rag['index'].search(q_emb, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        meta = rag['metadata'][idx]
        results.append({
            "score": float(scores[0][i]),
            "lang": meta["lang"],
            "question": meta["question"][:120],
            "id": meta["id"]
        })
    
    # Log metrics
    if track_metrics:
        mlflow.log_metric("top_score", results[0]["score"])
        mlflow.log_metric("avg_score", np.mean([r["score"] for r in results]))
        mlflow.log_metric("search_latency_ms", (time.time() - start_time) * 1000)
        st.sidebar.success(f"✅ Logged to MLflow: {run.info.run_id}")
    
    return results

def generate_answer(query, results):
    """Generate answer using retrieved QA pairs"""
    context = "\n".join([
        f"[{r['lang']}] Q: {r['question']} (score: {r['score']:.2f})" 
        for r in results[:3]
    ])
    
    prompt = f"""MLQA Knowledge Base:
{context}

User Question: {query}

Answer:"""
    
    result = rag['generator'](
        prompt, 
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=rag['generator'].tokenizer.eos_token_id
    )[0]['generated_text']
    
    return result[len(prompt):].strip()

# Execute search
if st.button("🚀 SEARCH & TRACK", type="primary", use_container_width=True) and query:
    with st.spinner("🔍 Searching MLQA knowledge base..."):
        results = search_mlqa(query, k_results)
        answer = generate_answer(query, results)
    
    # Results layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        lang = results[0]["lang"].upper()
        st.metric("Detected Language", lang)
    
    with col2:
        st.metric("Top Score", f"{results[0]['score']:.3f}")
    
    with col3:
        st.metric("Retrieved", len(results))
    
    # Detailed results
    st.subheader("📚 Top MLQA Matches")
    
    for i, result in enumerate(results):
        with st.expander(f"#{i+1} | {result['lang'].upper()} | {result['score']:.3f}"):
            st.write(f"**Q:** {result['question']}")
            st.caption(f"Doc ID: {result['id']}")
    
    # Generated answer
    st.subheader("🤖 Answer")
    st.info(answer)
    
    # Performance chart
    st.subheader("📊 Retrieval Scores")
    scores = [r["score"] for r in results]
    st.bar_chart(pd.DataFrame({"Rank": range(1, len(scores)+1), "Score": scores}))

# Instructions
with st.expander("📋 Quick Start"):
    st.markdown("""
    **1. Terminal 1:** `mlflow ui --port 5000`
    
    **2. Terminal 2:** `streamlit run app.py`
    
    **Test queries:**
    ```plaintext
    • "capital of India" (EN)
    • "भारत की राजधानी" (HI)  
    • "¿Quién ganó la guerra?" (ES)
    • "AIとは何ですか" (other langs)
    ```
    
    **MLflow:** http://localhost:5000
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Built with:** MLQA 5K Dataset • HuggingFace Transformers • MLflow • Streamlit"
)
