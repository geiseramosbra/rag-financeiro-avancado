import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from cohere import Client as CohereClient

# Carrega variáveis de ambiente
load_dotenv()

def buscar_contexto(pergunta):
    """
    Realiza a busca semântica no Qdrant, aplica Rerank com Cohere 
    e retorna o contexto otimizado para o LLM.
    """
    try:
        # Inicializa clientes com segurança
        cohere_client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60
        )
        
        # 1. Gera o vetor da pergunta
        embedding_response = cohere_client.embed(
            texts=[pergunta], 
            model="embed-multilingual-v3.0", 
            input_type="search_query"
        )
        vetor = embedding_response.embeddings[0]
        
        # 2. Busca inicial no Qdrant (Recuperação)
        resultados = qdrant_client.search(
            collection_name="finance_docs_cohere",
            query_vector=vetor,
            limit=10 
        )
        
        # Extrai os textos recuperados
        documentos = [r.payload.get("page_content", "") for r in resultados if r.payload]
        
        if not documentos:
            return "Nenhuma informação relevante encontrada no banco de dados."
        
        # 3. Rerank (Refinamento da busca)
        reranked = cohere_client.rerank(
            query=pergunta, 
            documents=documentos, 
            top_n=3, 
            model="rerank-v3.5"
        )
        
        # 4. Consolida o contexto para o LLM
        contexto = "\n---\n".join([documentos[r.index] for r in reranked.results])
        return contexto

    except Exception as e:
        return f"Erro ao processar a busca: {str(e)}"