from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from langchain.vectorstores.pgvector import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# --- Set environment variables ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyBQIKEIBPWZ_f7SQxJsLXkTnrW5fNcJAVA"

# --- Initialize FastAPI ---
app = FastAPI(title="Gemini RAG API")

# --- LangChain: Setup Embeddings and Vector Store ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

CONNECTION_STRING = "postgresql+psycopg2://languser:langpass@localhost:5432/langdb"
COLLECTION_NAME = "history_table"

vector_store = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# --- LangChain: Gemini Chat Model ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# --- Request Schema ---
class QueryRequest(BaseModel):
    query: str

# --- API Endpoint ---
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        # Step 1: Embed query and search similar chunks
        query = request.query
        docs = vector_store.similarity_search(query, k=10)
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        context = docs[0].page_content

        # Step 2: Ask Gemini LLM
        prompt = f"""You are an expert assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}"""
        response = llm([HumanMessage(content=prompt)])

        return {
            "query": query,
            "most_similar_chunk": context,
            "answer": response.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# this is the api ml    
