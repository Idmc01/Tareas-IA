import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Agente IA", page_icon="")
st.title("Asistente de Inteligencia Artificial - TEC")

client = OpenAI(api_key="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", base_url="https://api.openai.com/v1")

try:
    models = client.models.list()
    st.sidebar.success("Conexión con OpenAI establecida")
except Exception as e:
    st.sidebar.error(f"Error de conexión: {str(e)}")
    st.stop()

SYSTEM_PROMPT = """
Eres un asistente académico llamado AIDA, especializado en Inteligencia Artificial.
Tu tarea es responder preguntas sobre Inteligencia Artificial y temas relacionados.
Debes responder siempre de manera clara y con tono docente.
Si no tienes información suficiente para responder una pregunta, indícalo honestamente.
"""

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

        chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(st.session_state.messages)

        chat_completion = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=chat_messages)

        response_content = chat_completion.choices[0].message.content
        st.chat_message("assistant").markdown(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})

    except Exception as e:
        st.error(f"Error: {str(e)}")
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
