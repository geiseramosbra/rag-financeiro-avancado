import streamlit as st
import os
from dotenv import load_dotenv
from app import buscar_contexto
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="RAG Financeiro Avançado", page_icon="📊", layout="centered")

st.title("📊 Assistente IA - Relatórios 10-K")
st.write("Layout-Aware Chunking & Cohere Rerank ativos.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pergunta := st.chat_input("Ex: Quem deve assinar o relatório Form 10-K?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🔍 *Analisando blocos estruturados do PDF e aplicando Rerank...*")
        
        try:
            # 1. Recupera o contexto e as páginas fontes mapeadas
            contexto_real, paginas_fontes = buscar_contexto(pergunta)
            
            # 2. Monta o prompt rígido para a LLM
            prompt_sistema = (
                "Você é um especialista em análise financeira avançada. "
                "Responda à pergunta do usuário utilizando APENAS o contexto fornecido abaixo. "
                "Seja claro, objetivo, analítico e profissional. Se não encontrar a resposta, diga que não possui dados suficientes.\n\n"
                f"CONTEXTO:\n{contexto_real}"
            )
            
            # 3. Executa a chamada do Groq
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": pergunta}
                ],
                temperature=0.2
            )
            
            resposta_ia = completion.choices[0].message.content
            
            # 4. Formata a resposta visual adicionando as fontes encontradas de forma elegante
            resposta_final_com_fontes = (
                f"{resposta_ia}\n\n"
                f"--- \n"
                f"📖 **Fontes extraídas das páginas:** `{paginas_fontes}` do documento."
            )
            
            placeholder.markdown(resposta_final_com_fontes)
            st.session_state.messages.append({"role": "assistant", "content": resposta_final_com_fontes})
            
        except Exception as e:
            placeholder.markdown(f"❌ **Erro ao processar:** {str(e)}")