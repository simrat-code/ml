import json
import uuid
import os
import logging
import time
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

_logger.info("Loading environment variables from .env file...")
load_dotenv()  # Load environment variables from .env file


def load_homeopathy_json(file_path: str) -> List[Dict[str, Any]]:
    """Loads and parses the structured JSON file containing remedy details."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepare_text_for_embedding(remedy: Dict[str, Any]) -> str:
    """
    Formats key remedy information into a rich textual document chunk 
    optimized for dense vector embedding models.
    """
    symptoms = "; ".join(remedy.get("key_symptoms", []))
    modalities = remedy.get("modalities", {})
    worse = modalities.get("worse", "N/A")
    better = modalities.get("better", "N/A")
    
    comp = ", ".join(remedy.get("relationships", {}).get("complementary", [])) or "None"
    antidotes = ", ".join(remedy.get("relationships", {}).get("antidotes", [])) or "None"
    incomp = ", ".join(remedy.get("relationships", {}).get("incompatible", [])) or "None"

    doc_text = (
        f"Remedy: {remedy.get('name')} ({remedy.get('abbreviation', '')})\n"
        f"Key Symptoms: {symptoms}\n"
        f"Modalities - Aggravation: {worse} | Amelioration: {better}\n"
        f"Complementary Remedies: {comp}\n"
        f"Antidotes: {antidotes}\n"
        f"Incompatible Remedies: {incomp}"
    )
    return doc_text

def ingest_to_qdrant(
    client: QdrantClient,
    json_file_path: str,
    collection_name: str = "homeopathy_remedies",
    # qdrant_host: str = "localhost",
    # qdrant_port: int = 6333,
    # qdrant_api_key: str = None,
    model_name: str = "./models/all-MiniLM-L6-v2",
    use_in_memory: bool = True
):
    """
    Reads remedy JSON, vectorizes remedies using SentenceTransformer, 
    and upserts points into Qdrant vector database.
    """
    if not client:
        raise ValueError("Qdrant client must be provided.")
    
    print(f"Loading JSON data from: {json_file_path}")
    main_data = load_homeopathy_json(json_file_path)
    print(f"Loaded {len(main_data.get('remedies', []))} remedies from JSON.")
    remedies = main_data.get("remedies", [])

    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    vector_size = model.get_embedding_dimension()

    # Initialize Qdrant Client (In-memory mode for rapid local testing or connected instance)
    # if use_in_memory:
    #     client = QdrantClient(":memory:")
    # elif qdrant_api_key:
    #     client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
    # else:
    #     client = QdrantClient(host=qdrant_host, port=qdrant_port)

    print(f"qdrant collections: {client.get_collections()}")

    # Recreate or create collection
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created Qdrant collection '{collection_name}' (dim={vector_size}).")

    points = []
    print("Generating vector embeddings and preparing payload records...")
    for idx, remedy in enumerate(remedies):
        chunk_text = prepare_text_for_embedding(remedy)
        vector = model.encode(chunk_text).tolist()

        # Build Qdrant Point payload storing full structured fields for filtering / retrieval
        point = PointStruct(
            id=idx + 1,  # Can also use str(uuid.uuid4())
            vector=vector,
            payload={
                "remedy_id": remedy.get("id"),
                "name": remedy.get("name"),
                "abbreviation": remedy.get("abbreviation"),
                "key_symptoms": remedy.get("key_symptoms"),
                "modalities": remedy.get("modalities"),
                "complementary": remedy.get("relationships", {}).get("complementary"),
                "antidotes": remedy.get("relationships", {}).get("antidotes"),
                "incompatible": remedy.get("relationships", {}).get("incompatible"),
                "full_text_chunk": chunk_text
            }
        )
        points.append(point)

    # Upsert all points into Qdrant
    client.upsert(collection_name=collection_name, points=points)
    print(f"Successfully indexed {len(points)} remedies into Qdrant!")
    
    return client, model

def query_remedies(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    collection_name: str = "homeopathy_remedies",
    top_k: int = 3
):
    """
    Sample search function to test semantic symptom querying against Qdrant.
    """
    print(f"\n==================================================")
    print(f"Query: '{query}'")
    print(f"==================================================")
    
    query_vector = model.encode(query).tolist()
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )

    for i, point in enumerate(results.points, 1):
        p = point.payload
        print(f"\n[{i}] {p.get('name')} ({p.get('abbreviation')}) — Match Score: {point.score:.4f}")
        print(f"    Key Symptoms  : {'; '.join(p.get('key_symptoms', ['None']) or ['None'])}")
        print(f"    Worse / Better: Agg: {p.get('modalities', {}).get('worse')} | Amel: {p.get('modalities', {}).get('better')}")
        print(f"    Complementary : {', '.join(p.get('complementary', ['None']) or ['None'])}")
        print(f"    Incompatible  : {', '.join(p.get('incompatible', ['None']) or ['None'])}")
        print(f"    Antidotes     : {', '.join(p.get('antidotes', ['None']) or ['None'])}")

# Example execution test
if __name__ == "__main__":
    qdrant_host = os.getenv("qdrant_host", "localhost")
    qdrant_api_key = os.getenv("qdrant_api_key")
    qdrant_port = int(os.getenv("qdrant_port", 6333))

    collection_name = "homeopathy_remedies"
    model_name = "./models/all-MiniLM-L6-v2"

    if qdrant_api_key:
        client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
    else:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
    print(client.get_collections())
    time.sleep(1)  # Allow time for Qdrant to be ready

    print("Choose an option:")
    print("1. Create collection and index remedies")
    print("2. Run sample searches")
    print("3. Run a custom search")
    choice = input("Enter Choice: ").strip()

    if choice == "1":
        ingest_to_qdrant(
            client=client,
            json_file_path="homeopathy_remedies_summary.json",
            collection_name=collection_name,
            model_name=model_name,
            use_in_memory=False
        )
    elif choice in ("2", "3"):

        if not client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist. Choose option 1 first.")
        else:
            print(f"Loading embedding model: {model_name}...")
            model = SentenceTransformer(model_name)

            if choice == "2":
                sample_queries = [
                    "sudden fever with throbbing headache, high fever, red skin, and dilated pupils",
                    "extreme anxiety, acute restlessness, dry heat, and intense fear of death",
                    "joint pain worse from motion and better from rest or pressure",
                ]
                for sample_query in sample_queries:
                    query_remedies(client, model, query=sample_query, collection_name=collection_name, top_k=2)
            else:
                custom_query = input("Enter your search query: ").strip()
                if custom_query:
                    query_remedies(client, model, query=custom_query, collection_name=collection_name)
                else:
                    print("Search query cannot be empty.")
    else:
        print("Invalid choice. Run the program again and select 1, 2, or 3.")