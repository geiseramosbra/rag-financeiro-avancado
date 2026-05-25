import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from cohere import Client as CohereClient
from groq import Groq

load_dotenv()

COLLECTION_NAME = "finance_docs_cohere"

def buscar_contexto(pergunta_usuario):
    """Busca semântica no Qdrant + Reclassificação (Rerank) via Cohere"""
    
    # Inicializa os clientes aqui dentro para garantir que estejam disponíveis no Streamlit
    cohere_client = CohereClient(api_key=os.getenv("COHERE_API_KEY"))
    qdrant_client = QdrantClient(path="C:/Users/geise/Downloads/rag-financeiro-avancado/qdrant_db")
    
    # 1. Gera o embedding da pergunta
    response = cohere_client.embed(
        texts=[pergunta_usuario],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    vetor_pergunta = response.embeddings[0]
    
    # 2. Busca inicial estendida (10 chunks)
    resultados_iniciais = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vetor_pergunta,
        limit=10
    )
    
    # Mapeia os dados mantendo o texto e o metadado da página juntos
    dados_chunks = []
    lista_chunks_texto = []
    
    for ponto in resultados_iniciais.points:
        texto = ponto.payload["page_content"]
        pagina = ponto.payload.get("metadata", {}).get("page", "Desconhecida")
        
        dados_chunks.append({"texto": texto, "pagina": pagina})
        lista_chunks_texto.append(texto)
    
    # 3. Aplica o Rerank da Cohere nos textos
    rerank_response = cohere_client.rerank(
        query=pergunta_usuario,
        documents=lista_chunks_texto,
        top_n=3,
        model="rerank-v3.5"
    )
    
    # 4. Coleta os textos selecionados e mapeia suas respectivas páginas
    chunks_reclassificados = []
    paginas_fontes = set()
    
    for resultado in rerank_response.results:
        dados_originais = dados_chunks[resultado.index]
        chunks_reclassificados.append(dados_originais["texto"])
        paginas_fontes.add(str(dados_originais["pagina"]))
        
    contexto_formatado = "\n---\n".join(chunks_reclassificados)
    fontes_formatadas = ", ".join(sorted(list(paginas_fontes)))
    
    return contexto_formatado, fontes_formatadas

def responder_pergunta_com_rag(pergunta_usuario):
    """Função auxiliar para testes rápidos no terminal"""
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    contexto, fontes = buscar_contexto(pergunta_usuario)
    
    prompt_sistema = (
        "Você é um especialista em análise financeira avançada. "
        "Responda à pergunta do usuário utilizando APENAS o contexto fornecido abaixo.\n\n"
        f"CONTEXTO:\n{contexto}"
    )
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": pergunta_usuario}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.2
    )
    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    pergunta_teste = "Quem deve assinar o relatório Form 10-K em nome do registrador?"
    responder_pergunta_com_rag(pergunta_teste)