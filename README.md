# RAG Financeiro Avançado (Advanced Financial RAG)

[Português](#-português) | [English](#-english)

---

## 🇧🇷 Português

Este repositório contém um sistema **RAG (Retrieval-Augmented Generation) Avançado** de nível empresarial, projetado especificamente para analisar relatórios financeiros complexos (como o Form 10-K da SEC). 

O grande diferencial deste projeto não é apenas conversar com um PDF, mas garantir a **confiabilidade absoluta dos dados** e a **eliminação de alucinações**, problemas críticos no setor financeiro.

### Arquitetura e Tecnologias Utilizadas
O sistema utiliza uma arquitetura de dois estágios (*Two-Stage Retrieval*) para garantir máxima precisão:

1. **Layout-Aware Chunking (PyMuPDF / fitz):** Em vez de quebrar o PDF por número cego de caracteres (o que destrói tabelas e listas), o sistema analisa os blocos visuais do documento, mantendo tabelas e parágrafos estruturados de forma íntegra.
2. **Embeddings de Alta Performance (Cohere API):** Armazena o significado semântico profundo dos blocos de texto usando o modelo `embed-multilingual-v3.0`.
3. **Banco de Dados Vetorial (Qdrant Local):** Armazenamento em disco rápido e indexado para recuperação dos 10 blocos mais similares à pergunta do usuário.
4. **Rerank Avançado (Cohere Rerank v3.5):** A etapa crucial de produção. O algoritmo reclassifica os 10 blocos recuperados, filtra o "filé mignon" (os 3 mais relevantes de verdade) e descarta o ruído antes de enviar para a LLM.
5. **Geração de Resposta e Auditoria (Groq / Llama 3.1):** Gera respostas cirúrgicas baseadas *apenas* no contexto real e **injeta os metadados das páginas fontes**, permitindo que o usuário audite de qual página do PDF a informação foi extraída.
6. **Interface Gráfica (Streamlit):** Uma aplicação web limpa e intuitiva para o usuário final.

###  Como Executar o Projeto

1. Clone o repositório e instale as dependências:
   ```bash
   pip install qdrant-client cohere groq pymupdf streamlit python-dotenv