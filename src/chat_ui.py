import streamlit as st
import os
from app import buscar_contexto
from groq import Groq

st.set_page_config(page_title="RAG Financeiro", layout="centered")
st.title(" Assistente Financeiro 10-K")

if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Pergunte algo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            contexto = buscar_contexto(prompt)
            groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
            resposta = groq.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"Use este contexto para responder: {contexto}"}, 
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant"
            ).choices[0].message.content
            st.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})