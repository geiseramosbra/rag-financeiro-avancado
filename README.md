# Advanced Financial RAG System

This project is an advanced Retrieval-Augmented Generation (RAG) system designed to analyze financial reports (specifically Form 10-K documents). It leverages modern AI infrastructure to perform semantic search, document re-ranking, and context-aware question answering.



## Architecture Overview

The system follows a robust data pipeline:

* **Data Ingestion:** Extracts text from PDF documents using layout-aware chunking.
* **Embedding Generation:** Converts text into high-dimensional vectors using Cohere's multilingual embedding models.
* **Vector Database:** Stores and retrieves document chunks using Qdrant Cloud.
* **Retrieval & Reranking:** Implements a two-stage retrieval process with Cohere Rerank to ensure high-precision context delivery to the LLM.
* **Generation:** Uses Groq's Llama-3.1-8b-instant model to generate accurate answers based on the retrieved context.

## Technologies Used

* **Python:** Core programming language.
* **Streamlit:** Frontend interface for real-time interaction.
* **Cohere:** Embeddings and Re-ranking API.
* **Qdrant:** Vector database for semantic search.
* **Groq:** Inference engine for the Llama 3.1 LLM.
* **PyMuPDF (fitz):** PDF processing and text extraction.

## Setup and Installation

### Prerequisites

Ensure you have Python 3.10+ installed and the following API keys:

* Cohere API Key
* Qdrant Cloud URL and API Key
* Groq API Key

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd rag-financeiro-avancado