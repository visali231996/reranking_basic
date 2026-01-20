from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

# --------------------------
# 1. Load file
# --------------------------
loader = TextLoader("demo.txt", encoding="utf-8")
docs = loader.load()

# --------------------------
# 2. Split into chunks
# --------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# --------------------------
# 3. Embeddings + Vector DB
# --------------------------
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma.from_documents(chunks, emb)

# --------------------------
# 4. BGE RERANKER
# --------------------------
reranker = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-large"   # No device argument allowed
)

# --------------------------
# 5. LLM
# --------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template("""
Answer using the context:

{context}

Question: {question}
""")

# --------------------------
# 6. Query
# --------------------------
query = "problems in india"

# -------------------------------------------------------
# A. BEFORE RERANKER → Vector similarity scores
# -------------------------------------------------------
def get_response(query):
    print("TOP 8 BEFORE RERANKER (Similarity Scores)")
    initial_results = vectordb.similarity_search_with_score(query, k=8)
    initial_docs = []
    texts_for_reranker = []
    for i, (doc, score) in enumerate(initial_results):
        initial_docs.append(doc)
        texts_for_reranker.append(doc.page_content)

    print(f"\n--- Chunk {i+1} ---")
    print(f"Similarity Score: {score:.4f}")
    print(doc.page_content[:400])


# -------------------------------------------------------
# B. AFTER RERANKER → BGE relevance scores
# -------------------------------------------------------
    bge_reranker = CrossEncoder("BAAI/bge-reranker-large")

# Prepare pairs
    pairs = [[query, doc.page_content] for doc in initial_docs]

# Compute relevance scores
    scores = bge_reranker.predict(pairs)

# Sort docs by BGE score
    reranked = sorted(
        zip(scores, initial_docs),
        key=lambda x: x[0],
        reverse=True
)

# Print results

    print("TOP 8 AFTER BGE RERANKER")
    print("======================\n")
    for i, (score, doc) in enumerate(reranked):
        print(f"\n--- Reranked Chunk {i+1} ---")
        print(f"Relevance Score: {score:.4f}")
        print(doc.page_content[:400])


# -------------------------------------------------------
# C. FINAL Answer (RAG)
# -------------------------------------------------------

    print("FINAL ANSWER")
    print("======================\n")

    context_text = "\n\n".join([doc.page_content for _, doc in reranked])

    response = llm.invoke(
        prompt.format(context=context_text, question=query)
    )
    print(response.content)
    return response.content

get_response(query)


