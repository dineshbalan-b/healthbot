import os
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

EMBED_MODEL = "embed-english-light-v3.0"
LLM_MODEL = "llama-3.1-8b-instant"
RETRIEVER_K = 4
VECTOR_STORE_DIR = "data/faiss_index"

embeddings = None
vector_store = None
sessions = {}


def _get_embeddings():
    """Lazily create the Cohere embeddings object (requires COHERE_API_KEY at call time)."""
    global embeddings
    if embeddings is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY is not set. Please add it to your .env file.")
        embeddings = CohereEmbeddings(cohere_api_key=api_key, model=EMBED_MODEL)
    return embeddings


def init_vector_store():
    """Called at startup. Loads any existing FAISS index from disk (no crash if key is missing)."""
    global vector_store
    if os.path.exists(VECTOR_STORE_DIR):
        try:
            emb = _get_embeddings()
            vector_store = FAISS.load_local(VECTOR_STORE_DIR, emb, allow_dangerous_deserialization=True)
            print(" FAISS index loaded from disk.")
        except ValueError as e:
            # Missing API key — don't crash, just note it
            print(f"  {e}  — upload a PDF after setting COHERE_API_KEY to build the index.")
            vector_store = None
        except Exception as e:
            print(f" Could not load FAISS index: {e}")
            vector_store = None
    else:
        print("ℹ  No FAISS index found. Upload a PDF to get started.")
        vector_store = None


def get_vector_store():
    return vector_store


def add_documents_to_store(chunks: list):
    """Embed chunks and merge into the persistent FAISS index."""
    global vector_store
    emb = _get_embeddings()   # raises if API key missing

    texts = [c["text"] for c in chunks]
    metadatas = [{"title": c["title"]} for c in chunks]

    if vector_store is None:
        vector_store = FAISS.from_texts(texts, embedding=emb, metadatas=metadatas)
    else:
        new_vs = FAISS.from_texts(texts, embedding=emb, metadatas=metadatas)
        vector_store.merge_from(new_vs)

    # Persist to disk
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_DIR)
    print(f" FAISS index saved to '{VECTOR_STORE_DIR}'")
    # Reset sessions so the next chat picks up the updated retriever
    sessions.clear()


def get_rag_chain(session_id: str):
    if not vector_store:
        raise ValueError("Knowledge base not loaded. Please upload a PDF first.")

    if session_id not in sessions:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")

        llm = ChatGroq(temperature=0.0, model_name=LLM_MODEL, groq_api_key=groq_key)

        STRICT_PROMPT = PromptTemplate(
            template="""You are a STRICT and precise Document AI Assistant.
Your ONLY source of knowledge is the provided PDF document context below.
If the answer to the question is NOT explicitly found in the Context, you MUST refuse and reply:
"I am sorry, but I can only provide answers based on the uploaded document. Your query is not covered in the document."
Do NOT use your general knowledge or attempt to guess.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
        sessions[session_id] = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": STRICT_PROMPT}
        )

    return sessions[session_id]
