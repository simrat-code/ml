import os
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 1. Setup API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 2. Load your own local documents (e.g., Markdown, PDFs, or Text)
# Replace the path with your local documents directory
loader = DirectoryLoader("path/to/your/documents/", glob="**/*.md")
documents = loader.load()

# 3. Initialize the LLMs and Embeddings wrappers
# Ragas uses a generator model to write questions and a critic model to filter out bad ones
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# 4. Initialize the TestsetGenerator
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings
)

# 5. Generate the dataset 
# This returns an evaluation dataset containing question, contexts, and ground_truth
testset = generator.generate_with_langchain_docs(
    documents=documents,
    test_size=10,  # Number of evaluation pairs to generate
)

# 6. Export to a Pandas DataFrame or CSV
df = testset.to_pandas()
df.to_csv("synthetic_test_dataset.csv", index=False)

# Exported csv contains specific columns required for RAG evaluation
#   - question: the synthetically generated user query
#   - contexts: the actual chunks extracted from the docs used to build he question
#   - ground_truth: the ideal correct answer generated directly from the context.
#   - evolution_type: the type of question complexity applied eg multi-hop reasoning, conditional, simple.
