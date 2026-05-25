import os
import fitz  # PyMuPDF (Instalado via pip install pymupdf)
import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

load_dotenv()

# Inicializa o cliente oficial da Cohere usando a chave do seu .env
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

def extrair_chunks_por_layout(caminho_pdf):
    """
    Lê o PDF respeitando os blocos visuais e retorna uma lista de dicionários
    contendo o texto estruturado e o número da página de origem.
    """
    chunks_estruturados = []
    
    # Abre o documento usando o PyMuPDF
    doc = fitz.open(caminho_pdf)
    
    for idx_pagina, pagina in enumerate(doc):
        # Extrai o texto dividindo por blocos visuais (parágrafos, tabelas, títulos separados)
        blocos = pagina.get_text("blocks")
        
        texto_acumulado = ""
        for bloco in blocos:
            texto_bloco = bloco[4].strip() # O texto do bloco fica na posição 4 da tupla
            
            # Ignora cabeçalhos, rodapés ou blocos vazios com menos de 20 caracteres
            if len(texto_bloco) < 20:
                continue
                
            # Se o bloco atual estourar ~1200 caracteres, salvamos o acumulado e começamos outro
            if len(texto_acumulado) + len(texto_bloco) > 1200:
                chunks_estruturados.append({
                    "texto": texto_acumulado.strip(),
                    "pagina": idx_pagina + 1
                })
                texto_acumulado = texto_bloco + "\n"
            else:
                texto_acumulado += texto_bloco + "\n"
                
        # Adiciona o restante do texto da página se houver
        if texto_acumulado.strip():
            chunks_estruturados.append({
                "texto": texto_acumulado.strip(),
                "pagina": idx_pagina + 1
            })
            
    return chunks_estruturados

def obter_embeddings_reais(lista_de_textos):
    resposta = co.embed(
        texts=lista_de_textos,
        model="embed-multilingual-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    return resposta.embeddings.float

def executar_pipeline_ingestao():
    print("--- Iniciando Pipeline RAG Avançado (Layout-Aware Chunking) ---")
    
    pdf_path = os.path.join("data", "apple_10k.pdf")
    if not os.path.exists(pdf_path):
        print(f"Erro: O arquivo {pdf_path} não foi encontrado.")
        return

    print("\n[Passo 1 e 2] Extraindo e dividindo texto estruturado por layout...")
    chunks_com_metadados = extrair_chunks_por_layout(pdf_path)
    print(f"Sucesso: Criados {len(chunks_com_metadados)} chunks inteligentes com base nos blocos do PDF.")

    # Extrai apenas os textos para enviar para a API de Embeddings
    lista_textos = [c["texto"] for c in chunks_com_metadados]

    print("\n[Passo 3] Gerando Embeddings Reais via API da Cohere...")
    vetores = obter_embeddings_reais(lista_textos)
    print(f"Sucesso: {len(vetores)} vetores gerados com precisão semântica.")

    print("\n[Passo 4] Enviando vetores e metadados para o Qdrant...")
    path_qdrant = "C:/Users/geise/Downloads/rag-financeiro-avancado/qdrant_db"
    nome_colecao = "finance_docs_cohere"
    
    client = QdrantClient(path=path_qdrant)
    
    # Recria ou limpa a coleção para garantir que os dados antigos e mal fatiados saiam
    if client.collection_exists(collection_name=nome_colecao):
        client.delete_collection(collection_name=nome_colecao)
        
    client.create_collection(
        collection_name=nome_colecao,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    points = []
    for idx, (chunk_info, vetor) in enumerate(zip(chunks_com_metadados, vetores)):
        points.append(
            PointStruct(
                id=idx,
                vector=vetor,
                payload={
                    "page_content": chunk_info["texto"],
                    "metadata": {"page": chunk_info["pagina"]} # Guardando o número da página!
                }
            )
        )

    client.upsert(
        collection_name=nome_colecao,
        points=points
    )
    
    print(f"Sucesso absoluto! Dados estruturados salvos na coleção '{nome_colecao}'.")
    print("\n--- Pipeline concluído com sucesso! ---")

if __name__ == "__main__":
    executar_pipeline_ingestao()