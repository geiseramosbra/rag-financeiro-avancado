import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from cohere import Client as CohereClient

# Carrega variáveis do ambiente
load_dotenv()

def buscar_contexto(pergunta):
    # Inicializa clientes
    cohere_client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
    
    # Inicialização explícita do cliente Qdrant
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )
    
    # Gera o vetor de busca
    embedding_response = cohere_client.embed(
        texts=[pergunta], 
        model="embed-multilingual-v3.0", 
        input_type="search_query"
    )
    vetor = embedding_response.embeddings[0]
    
    # Busca no Qdrant aumentando o limite para 10 para maior cobertura
    resultados = qdrant_client.search(
        collection_name="finance_docs_cohere",
        query_vector=vetor,
        limit=10
    )
    
    # Extrai os textos (payload) garantindo que o campo existe
    documentos = [r.payload.get("page_content", "") for r in resultados if r.payload]
    
    if not documentos:
        return "Nenhum documento relevante foi encontrado no banco de dados."
    
    # Rerank para melhorar a precisão da resposta
    reranked = cohere_client.rerank(
        query=pergunta, 
        documents=documentos, 
        top_n=3, 
        model="rerank-v3.5"
    )
    
    # Monta o contexto final
    contexto = "\n---\n".join([documentos[r.index] for r in reranked.results])
    return contexto