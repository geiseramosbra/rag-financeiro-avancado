import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from cohere import Client as CohereClient

load_dotenv()

def buscar_contexto(pergunta):
    cohere_client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
    
    # Inicialização explícita
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    qdrant_client = QdrantClient(url=url, api_key=api_key)
    
    # Debug: Verifica se o método existe antes de chamar
    if not hasattr(qdrant_client, "search"):
        raise AttributeError(f"O objeto qdrant_client não possui o método 'search'. Tipo: {type(qdrant_client)}")

    vetor = cohere_client.embed(
        texts=[pergunta], 
        model="embed-multilingual-v3.0", 
        input_type="search_query"
    ).embeddings[0]
    
    resultados = qdrant_client.search(
        collection_name="finance_docs_cohere",
        query_vector=vetor,
        limit=5
    )
    
    documentos = [r.payload["page_content"] for r in resultados]
    
    reranked = cohere_client.rerank(
        query=pergunta, 
        documents=documentos, 
        top_n=3, 
        model="rerank-v3.5"
    )
    
    contexto = "\n---\n".join([documentos[r.index] for r in reranked.results])
    return contexto