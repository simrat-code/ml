import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import faiss
from typing import List, Optional
import os

class FAISSVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        
    def create_documents_from_dataframe(self, df: pd.DataFrame, text_column: str = None) -> List[Document]:
        try:
            # Handle different DataFrame structures
            if text_column is None:
                if len(df.columns) == 1:
                    text_column = df.columns[0]
                else:
                    text_column = df.columns[0]  # Use first column by default
                    print(f"Using column '{text_column}' as text source")
            
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
            # Filter out empty rows and ensure text length constraint
            df_filtered = df.copy()
            df_filtered[text_column] = df_filtered[text_column].dropna().astype(str)
            
            # Create Document objects
            documents = []
            for idx, row in df_filtered.iterrows():
                text = row[text_column].strip()
                                
                doc = Document(
                    page_content=text,
                    metadata={
                        'product': text.split('-')[0]
                    }
                )
                documents.append(doc)
            
            print(f"Created {len(documents)} documents from DataFrame")
            return documents
            
        except Exception as e:
            print(f"Error creating documents from DataFrame: {e}")
            return []
    
    def load_csv_data(self, csv_file_path: str, text_column: str = None) -> List[Document]:
        """
        Load data from CSV file and convert to LangChain Documents
        
        Args:
            csv_file_path: Path to the CSV file
            text_column: Column name containing text (if None, assumes single column or first column)
        
        Returns:
            List of Document objects
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Use the DataFrame method
            return self.create_documents_from_dataframe(
                df, 
                text_column, 
                additional_metadata={"source_file": csv_file_path}
            )
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return []
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        try:
            if not documents:
                print("No documents provided")
                return False

            # texts = [doc.page_content for doc in documents]
            # embeddings = self.embeddings.embed_documents(texts)
            # embedding_dim = len(embeddings[0])
            # print(f"embedding dim: {embedding_dim}")
            # index = faiss.IndexFlatIP(embedding_dim)
                
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings,
            )
            # self.vector_store = FAISS.from_embeddings(
            #     texts=texts,
            #     text_embeddings=embeddings,
            #     embedding=self.embeddings.embed_query,
            #     index=index
            # )
            
            print(f"Vector store created with {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False
    
    def add_documents(self, documents: List[Document]):
        """
        Add more documents to existing vector store
        
        Args:
            documents: List of Document objects to add
        """
        if self.vector_store is None:
            print("Vector store not initialized. Use create_vector_store first.")
            return
            
        try:
            self.vector_store.add_documents(documents)
            print(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            print(f"Error adding documents: {e}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents using cosine similarity
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of most similar documents
        """
        if self.vector_store is None:
            print("Vector store not initialized")
            return []
            
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search with similarity scores
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            print("Vector store not initialized")
            return []
            
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def save_vector_store(self, save_path: str):
        if self.vector_store is None:
            print("Vector store not initialized")
            return
            
        try:
            self.vector_store.save_local(save_path)
            print(f"Vector store saved to {save_path}")
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def load_vector_store(self, load_path: str):
        try:
            self.vector_store = FAISS.load_local(
                load_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from {load_path}")
        except Exception as e:
            print(f"Error loading vector store: {e}")

# Example usage
def main():
    # Initialize the vector store
    vs = FAISSVectorStore()
    
    # Example 1: Create documents directly from DataFrame
    print("=== Example 1: Creating documents from DataFrame ===")
    
    # Create sample DataFrame with multiple columns
    sample_data = [
        "Hello-world",
        "Good-morning",
        "How-are-you?",
        "Python-is-great",
        "Machine-learning-rocks",
        "AI-is-the-future",
        "Data-science-is-fun",
        "Vector-search-works",
        "Embeddings-are-useful",
        "Natural-language-processing"
    ]    
    
    df = pd.DataFrame(sample_data, columns=['tag'])
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create documents directly from DataFrame
    documents = vs.create_documents_from_dataframe(
        df, 
        text_column="tag"
    )
    
    # Create vector store
    if vs.create_vector_store(documents):
        print("Vector store created successfully!")
        
        # Perform similarity search
        query = "programming-language"
        print(f"\nSearching for: '{query}'")
        
        # Search with scores
        results = vs.similarity_search_with_score(query, k=3)
        
        print("\nTop 3 similar results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. Text: '{doc.page_content}' (Score: {score:.4f})")
            print(f"   Full metadata: {doc.metadata}")
            print()
        
        # Example 2: Filter and search by metadata
        print("=== Example 2: Searching with metadata filtering ===")
        
        # Get all documents and filter by category
        results = vs.similarity_search(query, k=3)  # Get all documents
        for doc in results:
            print(doc)
        
        
        # Example 3: Add more documents to existing store
        print("\n=== Example 3: Adding more documents ===")
        
        additional_data = pd.DataFrame(
            ["Deep-learning", "Neural-networks", "Computer-vision"],
            columns=['tag']
        )
        
        new_documents = vs.create_documents_from_dataframe(
            additional_data,
            text_column="tag"
        )
        
        vs.add_documents(new_documents)
        
        # Search again to see new results
        results2 = vs.similarity_search("neural", k=2)
        print(f"\nSearching for 'neural' after adding new documents:")
        for i, doc in enumerate(results2, 1):
            print(f"{i}. {doc.page_content} (Metadata: {doc.metadata})")
        
        # Save vector store
        vs.save_vector_store("./db/tags")
        print("\nVector store saved to ./db/tags")
        
        # Example 4: Load and test saved vector store
        print("\n=== Example 4: Loading saved vector store ===")
        vs_new = FAISSVectorStore()
        vs_new.load_vector_store("./db/tags")
        
        # Test loaded vector store
        print(f"Testing loaded vector store with query: 'artificial-intelligence'")
        results3 = vs_new.similarity_search_with_score("artificial- intelligence", k=3)
        for i, (doc, score) in enumerate(results3, 1):
            print(f"{i}. {doc.page_content} (Score: {score:.4f})")

def read_vector_store_example():
        vectorstore = FAISSVectorStore()
        # vectorstore.get_embedding_size()
        vectorstore.load_vector_store("./db/tags")
        # vectorstore.get_embedding_size()
        
        # Test loaded vector store
        print(f"Testing loaded vector store with query: 'artificial-intelligence'")
        results3 = vectorstore.similarity_search_with_score("artificial-intelligence", k=3)
        for i, (doc, score) in enumerate(results3, 1):
            print(f"{i}. {doc.page_content} ({score:.4f})")

        print(f"Testing loaded vector store with query: 'artificial-intelligence'")
        results3 = vectorstore.similarity_search("artificial-intelligence", k=3)
        for i, doc in enumerate(results3, 1):
            print(f"{i}. {doc.page_content} ")

if __name__ == "__main__":
    # main()
    read_vector_store_example()