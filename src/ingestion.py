import os
import fitz
import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

def extrair_chunks(caminho_pdf):
    doc = fitz.open(caminho_pdf)
    chunks = []
    for i, pagina in enumerate(doc):
        texto = pagina.get_text().strip()
        if len(texto) > 100:
            chunks.append({"text": texto, "page": i + 1})
    return chunks

def run_ingestion():
    # Caminho dinâmico para o PDF
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(BASE_DIR, "data", "apple_10k.pdf")
    
    chunks = extrair_chunks(pdf_path)
    texts = [c["text"] for c in chunks]
    embeddings = co.embed(texts=texts, model="embed-multilingual-v3.0", input_type="search_document").embeddings.float
    
    # Conexão via Nuvem (Qdrant Cloud)
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    
    if client.collection_exists("finance_docs_cohere"):
        client.delete_collection("finance_docs_cohere")
        
    client.create_collection(
        collection_name="finance_docs_cohere",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    
    points = [PointStruct(id=i, vector=v, payload={"page_content": c["text"], "metadata": {"page": c["page"]}}) 
              for i, (c, v) in enumerate(zip(chunks, embeddings))]
    
    client.upsert(collection_name="finance_docs_cohere", points=points)
    print("Ingestão concluída com sucesso na nuvem!")

if __name__ == "__main__":
    run_ingestion()