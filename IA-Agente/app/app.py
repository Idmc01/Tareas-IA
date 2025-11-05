import os
import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

st.set_page_config(page_title="Agente IA", page_icon="")
st.title("ChatMMI")

# Inicializar el cliente de OpenAI directamente para probar la conexión
client = OpenAI(
    api_key="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",
    base_url="https://api.openai.com/v1"
)

#Probar la conexión primero
try:
    models = client.models.list()
    st.sidebar.success("Conexión con OpenAI establecida")
except Exception as e:
    st.sidebar.error(f"Error de conexión: {str(e)}")
    st.stop()

SYSTEM_PROMPT = """
Eres un asistente académico llamado Astra, especializado en Inteligencia Artificial.
Tu tarea es responder preguntas sobre Inteligencia Artificial y temas relacionados.
Debes responder siempre de manera clara, directa y con tono formal.
Si no tienes información suficiente para responder una pregunta, indícalo.
"""

OPENAI_API_KEY = "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"

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo-0125",
    temperature=0.3,
    openai_api_base="https://api.openai.com/v1",
    client=client  
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Escribe tu pregunta...")

if user_input:
    try:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        #Construir la lista de mensajes incluyendo el historial en st.session_state
        chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(st.session_state.messages)

        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=chat_messages
        )
        
        response_content = chat_completion.choices[0].message.content
        st.chat_message("assistant").markdown(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})

    except Exception as e:
        st.error(f"Error: {str(e)}")
