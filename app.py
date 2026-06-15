import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from cohere import Client as CohereClient

load_dotenv()

def buscar_contexto(pergunta):
    # Inicializa clientes usando as variáveis de ambiente
    cohere_client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
    
    # Conexão com o Qdrant (ajustada para estabilidade na nuvem)
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False
    )
    
    # Gera o vetor de busca usando o modelo multilingual
    vetor = cohere_client.embed(
        texts=[pergunta], 
        model="embed-multilingual-v3.0", 
        input_type="search_query"
    ).embeddings[0]
    
    # Busca no Qdrant usando o método .search (estável e compatível)
    resultados = qdrant_client.search(
        collection_name="finance_docs_cohere",
        query_vector=vetor,
        limit=5
    )
    
    # Extrai os textos recuperados
    documentos = [r.payload["page_content"] for r in resultados]
    
    # Reordena os resultados para melhor precisão (Rerank)
    reranked = cohere_client.rerank(
        query=pergunta, 
        documents=documentos, 
        top_n=3, 
        model="rerank-v3.5"
    )
    
    # Monta o contexto final para o chat
    contexto = "\n---\n".join([documentos[r.index] for r in reranked.results])
    return contexto