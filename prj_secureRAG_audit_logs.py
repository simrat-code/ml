

import os
import transformers
import langchain
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from typing import List

# 1. Define the wrapper implementing LangChain's Embeddings interface
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # sentence-transformers returns numpy arrays; convert to float lists
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    
print(f"[=] Loading ONNX model for embeddings...")
raw_model = SentenceTransformer("./models/all-MiniLM-L6-v2-onnx", backend="onnx")  # Load the ONNX model for embeddings
embedding_model = SentenceTransformerEmbeddings(raw_model)

def get_logfile():
    """
    This function returns the path to the system log file.
    In a real-world scenario, this would point to the actual system log.
    For testing purposes, it points to a mock log file.
    """
    # log_path = "/var/log/syslog"  # Uncomment for real system logs
    log_path = os.path.join(os.getcwd(), "mcp", "mock", "syslog.txt")  # Mock log file for testing
    return log_path

def get_log_lines(log_path, last_n_lines=20):
    """
    This function reads the log file and returns the last n lines.
    """
    print(f"[=] reading file: {log_path}")
    if not os.path.exists(log_path):
        return []

    with open(log_path, "r") as f:
        lines = f.readlines()[-last_n_lines:]  # Last n lines for safety

    return [line.strip() for line in lines]

# def create_embeddings_for_logs(log_path, last_n_lines=20):
#     """
#     This function reads the log file and creates embeddings for each line.
#     It returns a list of tuples containing the log line and its corresponding embedding.
#     """
#     if not os.path.exists(log_path):
#         return []

#     with open(log_path, "r") as f:
#         lines = f.readlines()[-last_n_lines:]  # Last n lines for safety

#     embeddings = []
#     for line in lines:
#         embedding = embedding_model.encode(line.strip())
#         embeddings.append((line.strip(), embedding))

#     return embeddings

def create_vectorstore(text_list):
    """
    This function creates a Chroma vector store from a list of text entries.
    Each entry is embedded using the ONNX model and stored in the vector store.
    """
    if not text_list:
        return None

    # Create embeddings for the text list
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    # convert to Document() objects for Chroma
    docs = text_splitter.create_documents(text_list)
    print(f"[=] total docs: {len(docs)}")

    # Create a Chroma vector store
    print(f"[=] initializing chroma vector store...")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)

    return vectorstore

def append_to_vectorstore(vectorstore, new_text):
    """
    This function appends new text to an existing Chroma vector store.
    It creates embeddings for the new text and adds them to the vector store.
    """
    if not vectorstore or not new_text:
        return vectorstore

    # Create embeddings for the new text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.create_documents([new_text])
    print(f"[=] new docs: {len(docs)}")

    # Add new documents to the existing vector store
    vectorstore.add_documents(docs)

    return vectorstore

def ask_logs(question, vectorstore):
    """
    This function takes a question and a Chroma vector store, retrieves relevant log entries,
    and returns the most relevant log lines based on the question.
    """
    if not vectorstore or not question:
        return []

    # Retrieve relevant documents from the vector store
    relevant_docs = vectorstore.similarity_search(question, k=5)  # Retrieve top 5 relevant docs

    # Extract the text from the retrieved documents
    relevant_logs = "\n".join([doc.page_content for doc in relevant_docs])

    return relevant_logs

if __name__ == "__main__":
    log_path = get_logfile()
    log_lines = get_log_lines(log_path)
    # embeddings = create_embeddings_for_logs(log_path)
    # print(f"[=] Created embeddings for {len(embeddings)} log lines.")

    # Create a vector store from the log lines
    # log_lines = [line for line, _ in embeddings]
    vectorstore = create_vectorstore(log_lines)
    if vectorstore:
        print("[=] Vector store created successfully.")
        ask_question = "What are the recent errors in the system logs?"
        relevant_logs = ask_logs(ask_question, vectorstore)
        print(f"[=] Relevant logs for question '{ask_question}':\n{relevant_logs}")
    else:
        print("[!] Failed to create vector store.")

