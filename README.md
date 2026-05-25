# RAG Financeiro Avancado (Advanced Financial RAG)

[Portugues](#portugues) | [English](#english)

---

## Portugues

Este repositorio contem um sistema RAG (Retrieval-Augmented Generation) de nivel empresarial, projetado especificamente para analisar relatorios financeiros complexos (como o Form 10-K da SEC). 

O diferencial deste projeto e garantir a confiabilidade absoluta dos dados e a eliminacao de alucinacoes.

### Arquitetura e Tecnologias
1. Layout-Aware Chunking: Uso de PyMuPDF para processar PDFs mantendo a integridade de tabelas e listas.
2. Embeddings: Modelo embed-multilingual-v3.0 da Cohere.
3. Banco de Dados Vetorial: Qdrant para armazenamento e busca eficiente.
4. Rerank Avancado: Algoritmo de reclassificacao Cohere Rerank v3.5 para filtrar ruido e garantir precisao.
5. LLM: Uso do Groq com Llama 3.1 para respostas auditativeis (com indicacao de pagina fonte).

### Como Executar
1. Clone o repositorio e instale as dependencias:
   pip install qdrant-client cohere groq pymupdf streamlit python-dotenv
2. Configure seu arquivo .env com suas chaves (use o .env.example como guia).
3. Execute a ingestao: python src/ingestion.py
4. Inicie a interface: streamlit run src/interface.py

---

## English

This repository contains a production-ready Advanced RAG (Retrieval-Augmented Generation) system designed specifically to analyze complex financial reports (such as SEC Form 10-K).

The core value of this project is ensuring absolute data reliability and eliminating hallucinations.

### Architecture & Technologies
1. Layout-Aware Chunking: Using PyMuPDF to process PDFs while preserving tables and list structures.
2. Embeddings: Powered by Cohere's embed-multilingual-v3.0 model.
3. Vector Database: Qdrant for fast and indexed semantic storage.
4. Advanced Reranking: Utilizing Cohere Rerank v3.5 to filter noise and ensure pinpoint accuracy.
5. LLM: Groq & Llama 3.1 for analytical answers with source page citation.

### How to Run
1. Clone the repository and install dependencies:
   pip install qdrant-client cohere groq pymupdf streamlit python-dotenv
2. Set up your .env file (see .env.example for the structure).
3. Run the ingestion process: python src/ingestion.py
4. Launch the interface: streamlit run src/interface.py