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
    
    # --- MODIFICAÇÃO PARA DIAGNÓSTICO ---
    # Usando scroll para ler diretamente o conteúdo do banco, 
    # ignorando a similaridade vetorial da pergunta.
    resultados, _ = qdrant_client.scroll(
        collection_name="finance_docs_cohere",
        limit=10,
        with_payload=True
    )
    
    # Extrai os textos do payload
    documentos = [r.payload.get("page_content", "Conteúdo vazio") for r in resultados if r.payload]
    
    if not documentos:
        return "Nenhum documento encontrado ou conteúdo vazio no banco."
    
    # Monta o texto para exibir na tela do Streamlit
    contexto = "\n\n--- DADOS NO BANCO ---\n\n".join(documentos)
    return contexto