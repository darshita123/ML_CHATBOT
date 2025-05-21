import os
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBQIKEIBPWZ_f7SQxJsLXkTnrW5fNcJAVA"

# Load text file and parse by pages
with open(r"output.txt", "r",encoding = "utf-8") as f:
    raw_text = f.read()

pages = re.findall(r"page\s*:\s*(\d+).*?content\s*:(.*?)=+", raw_text, re.DOTALL)

# Initialize Google Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# PostgreSQL connection string
CONNECTION_STRING = "postgresql+psycopg2://languser:langpass@localhost:5432/langdb"

# Create PGVector store
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name="gemini_table",
    embedding_function=embeddings,
)

# Text Splitter (if needed)
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Prepare Documents
documents = []
for page_num, content in pages:
    chunks = splitter.split_text(content.strip())
    for chunk in chunks:
        documents.append(
            Document(page_content=chunk, metadata={"page": int(page_num)})
        )

# Add documents to the vector store
vectorstore.add_documents(documents)
print("✅ Documents embedded and stored in PostgreSQL.")
