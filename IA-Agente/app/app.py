import os
import streamlit as st
from openai import OpenAI
from tools.web_search import search_web
from typing import List, Dict

from tools.rag import load_index_and_df, retrieve_from_index

#----- intentar cargar .env
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    print("No se pudo cargar el archivo .env!")


#----- Configuración de la página
st.set_page_config(page_title="Agente IA", page_icon="")
st.title("Asistente de Inteligencia Artificial - TEC")

#----- leer API_KEY desde .env
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    st.sidebar.error("Falta API_KEY!")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url="https://api.openai.com/v1")


#----- probar conexión con OpenAI
try:
    models = client.models.list()
    st.sidebar.success("Conexión con OpenAI establecida")
except Exception as e:
    st.sidebar.error(f"Error de conexión: {str(e)}")
    st.stop()

#----- prompt del sistema
SYSTEM_PROMPT = """
Eres un asistente académico llamado AIDA, especializado en Inteligencia Artificial.
Tu tarea es responder preguntas sobre Inteligencia Artificial y temas relacionados.
Debes responder siempre de manera clara y con tono docente.
Si no tienes información suficiente para responder una pregunta, indícalo honestamente.
"""

#----- Inicializar estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

#----- CONTROLES EN SIDEBAR
st.sidebar.markdown("---")
# RAG
st.sidebar.subheader("RAG (Apuntes del curso)")
rag_enabled = st.sidebar.checkbox("Usar RAG con base vectorial", value=True)
rag_strategy_label = st.sidebar.radio(
    "Fuente de RAG",
    ["Sliding window", "Recursive"],
    index=0,
    disabled=not rag_enabled,
)
rag_strategy = "sliding" if rag_strategy_label.startswith("Sliding") else "recursive"
rag_k = st.sidebar.slider("Resultados RAG (k)", min_value=1, max_value=10, value=5, disabled=not rag_enabled)

# Intento de precarga para informar estado

# Precarga para informar estado
if rag_enabled:
    try:
        idx_tmp, df_tmp, text_col_tmp, meta_tmp = load_index_and_df(rag_strategy)
        st.sidebar.success(f"Índice RAG cargado: {idx_tmp.ntotal} vectores")
    except Exception as e:
        st.sidebar.warning(f"No se pudo cargar el índice RAG: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Configuración de Búsqueda Web")
use_web = st.sidebar.checkbox("Buscar en la web antes de responder", value=False)
web_limit = st.sidebar.slider("Número de resultados web", min_value=1, max_value=10, value=3)

#----- Mostrar mensajes anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#----- Input del usuario
user_input = st.chat_input("Escribe tu pregunta...")

if user_input:
    # Mostrar mensaje del usuario
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    web_context = None
    rag_context = None
    rag_sources: List[Dict] = []

    # RAG (búsqueda en apuntes)
    if rag_enabled:
        try:
            with st.spinner("Buscando en apuntes (RAG)..."):
                rag_sources, rag_context = retrieve_from_index(client, user_input, rag_strategy, rag_k)
        except Exception as e:
            st.warning(f"RAG desactivado por error: {e}")
            rag_context = None
            rag_sources = []

        # mostrar resultados RAG
        if rag_sources:
            with st.expander("Resultados RAG (apuntes)", expanded=True):
                for r in rag_sources:
                    st.markdown(f"**[{r['rank']}]** Score: {r['score']:.4f} | Fuente: {r['fuente']} | Autor: {r['autor']} | Semana: {r['semana']}")
                    st.markdown(r["preview"])
                    st.markdown("---")

    # Buscar en la web si está activado
    if use_web:
        try:
            with st.spinner("Buscando en la web..."):
                results = search_web(user_input, limit=web_limit)
        except RuntimeError as e:
            st.warning(str(e))
            results = []
        except Exception as e:
            st.error(f"Error buscando en la web: {e}")
            results = []

        # Mostrar resultados web
        if not results:
            st.info("No se encontraron resultados en la web para la consulta.")
        else:
            with st.expander("Resultados de la búsqueda web", expanded=False):
                for i, r in enumerate(results, start=1):
                    st.markdown(f"**{i}. {r.get('title', 'Sin título')}**")
                    if r.get('snippet'):
                        st.markdown(r.get('snippet'))
                    if r.get('url'):
                        st.markdown(f"[{r.get('url')}]({r.get('url')})")
                    st.markdown("---")

            # Construir contexto para el modelo
            lines = []
            for i, r in enumerate(results, start=1):
                title = r.get('title', '')
                snippet = r.get('snippet', '')
                url = r.get('url', '')
                lines.append(f"{i}. {title}\n{snippet}\nFuente: {url}\n")
            web_context = "\n".join(lines)

    # Construir mensajes del chat
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if rag_context:
        chat_messages.append({
            "role": "system",
            "content": (
                "Contexto documental (apuntes del curso). Usa este contenido como fuente principal para responder.\n"
                "Incluye citas en el formato [n] cuando corresponda y no inventes fuentes.\n\n"
                + rag_context
            ),
        })

    if web_context:
        chat_messages.append({
            "role": "system",
            "content": "Información obtenida en la web (usar como referencia y citar fuentes cuando sea posible):\n\n" + web_context,
        })

    chat_messages += st.session_state.messages

    # generar respuesta
    try:
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                chat_completion = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=chat_messages
                )

                response_content = chat_completion.choices[0].message.content
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})

                # Mostrar fuentes RAG debajo (si existen)
                if rag_sources:
                    st.markdown("\n**Fuentes citadas (RAG):**")
                    for r in rag_sources:
                        st.markdown(f"- [{r['rank']}] {r['fuente']} — Autor: {r['autor']} — Semana: {r['semana']}")

    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")