# Advanced Financial RAG with LLMOps


This repository contains an enterprise-grade RAG (Retrieval-Augmented Generation) system, specifically engineered to analyze complex financial reports, such as SEC Form 10-K filings.

The core differentiator of this project is ensuring absolute data reliability through an integrated LLMOps pipeline based on the LLM-as-a-Judge framework, which automatically mitigates and eliminates hallucinations.

### Architecture and Technologies

* **Layout-Aware Chunking:** Utilizes PyMuPDF to process PDF documents while preserving the structural integrity of financial tables and lists.
* **Embeddings:** Powered by Cohere's embed-multilingual-v3.0 model.
* **Vector Database:** Leverages Qdrant for efficient semantic storage and indexing.
* **Advanced Reranking:** Integrates the Cohere Rerank v3.5 algorithm to filter background noise and maximize contextual precision.
* **LLM & LLMOps:** Uses Llama 3.3 via Groq Cloud for both response generation and automated auditing, enforced by strict data validation with Pydantic and JsonOutputParser.

### Automated Evaluation Pipeline (LLM-as-a-Judge)

Managed entirely by the `pipeline_avaliacao.py` script, the evaluation loop executes:
* **Phase 1 (Ground Truth):** Scans real contexts stored within the Qdrant database to automatically synthesize idealized question-and-answer pairs.
* **Phase 2 (Auditing):** Passes the test cases through the RAG pipeline and activates an independent AI Judge (running Llama 3.3 at temperature 0.0) to assess factual faithfulness, outputting structured technical reports in JSON format.

### How to Run

1. Install the project dependencies:
```bash
pip install qdrant-client cohere groq pymupdf streamlit python-dotenv langchain-core langchain-groq pydantic