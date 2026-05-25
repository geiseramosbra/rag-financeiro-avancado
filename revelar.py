import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

print("\n--- COLEÇÕES EXISTENTES NO SEU QDRANT ---")
print([c.name for c in qdrant_client.get_collections().collections])
print("-----------------------------------------\n")