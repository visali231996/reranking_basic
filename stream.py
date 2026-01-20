import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

st.title("🔍 RAG + BGE Reranker Demo")

# ------------------------------------
# Load + Split Documents
# ------------------------------------
loader = TextLoader("demo.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# ------------------------------------
# Embeddings + Vector DB
# ------------------------------------
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma.from_documents(chunks, emb)

# ------------------------------------
# Cross Encoder (BGE Reranker)
# ------------------------------------
bge_reranker = CrossEncoder("BAAI/bge-reranker-large")

# ------------------------------------
# LLM
# ------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template("""
Use the context below to answer the question:

{context}

Question: {question}
""")


# ------------------------------------
# Streamlit UI
# ------------------------------------
st.set_page_config(layout="wide")



query = st.text_input("Enter your query:")

if st.button("Run RAG Search"):
    # ------------------------------------
    # A. BEFORE RERANKER
    # ------------------------------------
    initial_results = vectordb.similarity_search_with_score(query, k=8)

    initial_docs = [doc for doc, _ in initial_results]
    similarity_scores = [score for _, score in initial_results]

    # ------------------------------------
    # B. AFTER RERANKER
    # ------------------------------------
    pairs = [[query, doc.page_content] for doc in initial_docs]
    bge_scores = bge_reranker.predict(pairs)

    reranked = sorted(
        zip(bge_scores, initial_docs),
        key=lambda x: x[0],
        reverse=True
    )

    reranked_scores = [s for s, _ in reranked]
    reranked_docs = [d for _, d in reranked]

    # ---------------------------------------------------
    # DISPLAY RESULTS IN STREAMLIT
    # ---------------------------------------------------
    col1, col2 = st.columns(2)

    # -----------------------
    # Left Column: BEFORE RERANKING
    # -----------------------
    with col1:
        st.subheader("🔴 Before Reranking (Vector Similarity)")
        for i, (score, doc) in enumerate(zip(similarity_scores, initial_docs)):
            st.markdown(f"### Chunk {i+1}")
            st.text(f"Similarity Score: {score:.4f}")
            st.write(doc.page_content)

    # -----------------------
    # Right Column: AFTER RERANKING
    # -----------------------
    with col2:
        st.subheader("🟢 After Reranking (BGE Relevance)")
        for i, (score, doc) in enumerate(zip(reranked_scores, reranked_docs)):
            st.markdown(f"### Reranked Chunk {i+1}")
            st.text(f"Relevance Score: {score:.4f}")
            st.write(doc.page_content)

    # ---------------------------------------------------
    # FINAL ANSWER
    # ---------------------------------------------------
    st.subheader("🔵 Final Generated Answer")

    context_text = "\n\n".join([d.page_content for d in reranked_docs])

    final_response = llm.invoke(
        prompt.format(context=context_text, question=query)
    )

    st.success(final_response.content)
