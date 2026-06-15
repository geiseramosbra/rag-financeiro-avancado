import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from cohere import Client as CohereClient

load_dotenv()

def buscar_contexto(pergunta):
    cohere_client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
    
    # Conexão via Nuvem ajustada para maior estabilidade
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False, 
        check_compatibility=False
    )
    
    vetor = cohere_client.embed(texts=[pergunta], model="embed-multilingual-v3.0", input_type="search_query").embeddings[0]
    
    resultados = qdrant_client.query_points(collection_name="finance_docs_cohere", query=vetor, limit=5).points
    documentos = [p.payload["page_content"] for p in resultados]
    
    reranked = cohere_client.rerank(query=pergunta, documents=documentos, top_n=3, model="rerank-v3.5")
    contexto = "\n---\n".join([documentos[r.index] for r in reranked.results])
    return contexto