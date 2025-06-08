import os
from typing import List
import urllib.request
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# prepare data
URL_PREFIX = "https://learn.microsoft.com/en-us/azure/machine-learning/"
URL_NAME_LIST = [
    "tutorial-azure-ml-in-a-day",
    "overview-what-is-azure-machine-learning",
    "concept-v2",
]

def get_file_chunks(file_name: str) -> List[str]:
    with open(file_name, "r", encoding="utf-8") as f:
        page_content = f.read()
        # use BeautifulSoup to parse HTML content
        soup = BeautifulSoup(page_content, "html.parser")
        text = soup.get_text(" ", strip=True)
        chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
        return chunks

local_file_path = os.path.join(os.getcwd(), "data")
# os.makedirs(local_file_path, exist_ok=True)
# for url_name in URL_NAME_LIST:
#     url = os.path.join(URL_PREFIX, url_name)
#     destination_path = os.path.join(local_file_path, url_name)
#     urllib.request.urlretrieve(url, destination_path)


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

index = 0
for root, _, files in os.walk(local_file_path):
    for file in files:
        index += 1
        print(f"processing: {file}")
        each_file_path = os.path.join(root, file)

        # Split the file into chunks.
        chunks = get_file_chunks(each_file_path)

        # vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store = FAISS.from_texts(chunks, embeddings)
        print(f"saving: ./db/faiss_index_{index}")
        vector_store.save_local(f"./db/faiss_index_{index}")

#         count = len(chunks)
#         if URL_PREFIX is not None:
#             metadatas = [
#                 {"title": file, "source": os.path.join(URL_PREFIX, file)}
#             ] * count
#         else:
#             metadatas = [{"title": file}] * count

#         # Embed chunks into embeddings, generate index in embedding store.
#         # If your data is large, inserting too many chunks at once may cause
#         # rate limit error，you can refer to the following link to find solution
#         # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quotas-limits
#         store.batch_insert_texts(chunks, metadatas)
#         print(f"Create index for {file} file successfully.\n")