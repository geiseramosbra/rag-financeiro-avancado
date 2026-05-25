import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def obter_embedding_mock(texto):
    import random
    random.seed(hash(texto))
    return [random.uniform(-1, 1) for _ in range(384)]

def buscar_no_rag(pergunta, top_k=3):
    path_qdrant = "./qdrant_db"
    nome_colecao = "finance_docs"
    
    client = QdrantClient(path=path_qdrant)
    vetor_pergunta = obter_embedding_mock(pergunta)
    
    # Substituído .search() por .query_points() que é o método correto na versão atual do Qdrant
    resposta = client.query_points(
        collection_name=nome_colecao,
        query=vetor_pergunta,
        limit=top_k
    )
    
    print(f"\n=== Resultados encontrados para: '{pergunta}' ===")
    for i, res in enumerate(resposta.points, 1):
        print(f"\n[Resultado {i}] (Score de Similaridade: {res.score:.4f})")
        print(f"Trecho: {res.payload.get('page_content')}")
        print("-" * 50)

if __name__ == "__main__":
    termo_busca = "Apple" 
    buscar_no_rag(termo_busca)