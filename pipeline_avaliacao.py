import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Inicialização dos modelos
llm_gerador = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
llm_juiz = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

class ParPerguntaResposta(BaseModel):
    pergunta: str = Field(description="Uma pergunta complexa baseada estritamente no contexto fornecido.")
    resposta_esperada: str = Field(description="A resposta ideal e detalhada para a pergunta, baseada no contexto.")

parser = JsonOutputParser(pydantic_object=ParPerguntaResposta)

print("INFO: Extraindo amostras de contextos reais do banco Qdrant...")
chunks_reais = []

# Gerenciamento do ciclo de vida do QdrantClient apontando para a sua pasta real
qdrant_path = os.path.join(os.getcwd(), "qdrant_db")
client_qdrant = QdrantClient(path=qdrant_path)

try:
    # Definindo a sua coleção real que vimos na estrutura de pastas
    nome_colecao = "finance_docs"
    
    # Realiza o scroll para buscar dados reais do relatório da Apple
    registros, _ = client_qdrant.scroll(
        collection_name=nome_colecao,
        limit=3,
        with_payload=True,
        with_vectors=False
    )
    
    if not registros:
        raise ValueError(f"Coleção '{nome_colecao}' encontrada, mas está sem registros.")
        
    for reg in registros:
        payload = reg.payload
        # Mapeamento estrito para o padrão de payloads do LangChain
        texto = payload.get("page_content") or payload.get("text") or payload.get("content")
        if texto:
            chunks_reais.append(texto)

except Exception as e:
    print(f"AVISO: Falha ao ler dados reais do Qdrant ({e}). Utilizando fallback corporativo.")
    chunks_reais = [
        "In fiscal year 2025, Nvidia's Data Center revenue expanded by 112% year-over-year, reaching a record $47.5 billion, driven by Hopper architecture.",
        "In 2025, Microsoft's Cloud revenue increased by 22% year-over-year to $115 billion, driven by strong growth in Azure.",
        "Alphabet reported Google Services revenues of $76.5 billion in Q4 2025, representing a 13% increase driven by Search and YouTube."
    ]
finally:
    client_qdrant.close()

dataset_testes = []

print("\nFase 1: Gerando Dataset Sintético de Avaliação (Ground Truth)...")
prompt_gerador = ChatPromptTemplate.from_template(
    "Você é um auditor financeiro sênior especializado em relatórios SEC 10-K.\n"
    "Com base APENAS no contexto abaixo, gere uma pergunta complexa de negócios e a resposta ideal esperada.\n\n"
         "CONTEXTO:\n{contexto}\n\n"
    "{format_instructions}"
)
chain_gerador = prompt_gerador | llm_gerador | parser

for i, chunk in enumerate(chunks_reais):
    try:
        resultado = chain_gerador.invoke({"contexto": chunk, "format_instructions": parser.get_format_instructions()})
        resultado["contexto_original"] = chunk
        dataset_testes.append(resultado)
        print(f"Sucesso: Caso de teste {i+1} gerado.")
    except Exception as e:
        print(f"Erro no chunk {i+1}: {e}")

print("\nFase 2: Iniciando Avaliação Automatizada em Lote (LLM-as-a-Judge)...")

prompt_rag = ChatPromptTemplate.from_template(
    "Responda à pergunta do usuário baseando-se no contexto fornecido.\n\nContexto: {contexto}\n\nPergunta: {pergunta}"
)
chain_rag = prompt_rag | llm_juiz

prompt_avaliacao = ChatPromptTemplate.from_template(
    "Você é um especialista em garantia de qualidade (QA) de sistemas de IA.\n"
    "Analise a RESPOSTA DO RAG comparando-a estritamente com o CONTEXTO ORIGINAL e com o GABARITO.\n\n"
    "CONTEXTO ORIGINAL: {contexto}\n"
    "GABARITO ESPERADO: {gabarito}\n"
    "RESPOSTA DO RAG: {resposta_rag}\n\n"
    "Responda estritamente no formato JSON abaixo:\n"
    "{{\n  \"nota\": [Insira apenas o número de 0 a 10],\n  \"justificativa\": \"[Sua justificativa técnica]\"\n}}"
)
chain_avaliador = prompt_avaliacao | llm_juiz

relatorio_final = []

for idx, teste in enumerate(dataset_testes):
    resposta_rag = chain_rag.invoke({"contexto": teste["contexto_original"], "pergunta": teste["pergunta"]}).content
    
    resultado_auditoria_bruto = chain_avaliador.invoke({
        "contexto": teste["contexto_original"],
        "gabarito": teste["resposta_esperada"],
        "resposta_rag": resposta_rag
    }).content
    
    try:
        dados_auditoria = json.loads(resultado_auditoria_bruto)
        relatorio_final.append({
            "id": idx + 1,
            "pergunta": teste["pergunta"],
            "nota_fidelidade": dados_auditoria.get("nota"),
            "justificativa": dados_auditoria.get("justificativa")
        })
    except:
        print(f"Erro na conversão do resultado JSON do caso {idx+1}")

print("\n==============================================")
print("SUMÁRIO EXECUTIVO DE QUALIDADE - LLMOPS")
print("==============================================")
for item in relatorio_final:
    print(f"Caso {item['id']} | Nota de Fidelidade: {item['nota_fidelidade']}")
    print(f"Justificativa: {item['justificativa']}\n----------------------------------------------")