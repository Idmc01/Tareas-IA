import os
import streamlit as st
from openai import OpenAI
from typing import List, Dict
from pathlib import Path

from rag_tool import load_index_and_df, retrieve_from_index
from web_tool import search_web

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    print("No se pudo cargar el archivo .env!")


# conf de pag
st.set_page_config(page_title="Agente IA", page_icon="")
st.title("ChatMMI")

# api_key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.sidebar.error("Falta API_KEY")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url="https://api.openai.com/v1")
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
Cuando la información solicitada exista en los apuntes o documentos disponibles, 
debes basar tu respuesta en ellos e indicar el documento y autor de referencia. 
Si no se encuentra información relevante en los apuntes, 
debes indicarlo explícitamente antes de responder con tu conocimiento general, 
manteniendo siempre la precisión y el tono académico.
"""
import re
def clean_search_query(query: str) -> str:
    #lista de palabras y frases comunes a eliminar
    stopwords = [
        'dame', 'dime', 'que es', 'qué es', 'cual es', 'cuál es', 'como es', 'cómo es',
        'cuales son', 'cuáles son', 'como son', 'cómo son', 'me puedes', 'puedes',
        'explicame', 'explícame', 'explica', 'dime sobre', 'dame información',
        'quiero saber', 'necesito saber', 'me gustaría', 'quisiera',
        'por favor', 'gracias', 'hola', 'ayudame', 'ayúdame', "quien es", "quien fue", "quién es", "quién fue", 
        "¿", 'que','qué','cual','cuál','cuales','cuáles','cuando','cuándo','donde',
        'dónde','por que','por qué','porque','para que','para qué','como','cómo',
        'quien','quién','quienes','quiénes','cuanto','cuánto','cuantos','cuántos',
        'cuanta','cuánta','cuantas','cuántas','si puedes','si pudieras','si fuese posible','seria posible',
        'sería posible','te agradecería','agradeceria','agradecería','gracias',
        'muchas gracias','hola','buenas','buenas tardes','buenas noches','buen día',
        'buenos dias','buenos días','saludos','estimado','dias','días','noches',"tardes",
        'estimado','queria','quería','quisiera saber','me gustaria saber',
        'me gustaría saber','podrias explicar','podrías explicar',
        'podrias decirme','podrías decirme','podrias darme','podrías darme','ahora','busca', '¿Que es', '¿Qué es', 
        '¿Cual es', '¿Cuál es', '¿Como es', '¿Cómo es', '¿Cuales son', '¿Cuáles son', '¿Como son', '¿Cómo son', 
        '¿Me puedes', '¿Puedes', '¿Explicame', '¿Explícame', '¿Explica', '¿Dime sobre', '¿Dame información', 
        '¿Quiero saber', '¿Necesito saber'
    ]

    query_lower = query.lower()
    cleaned = query

    #remove stopwords
    # eliminar todas las frases stop sin importar posición
    for phrase in stopwords:
        # usa regex para eliminar la frase completa 
        pattern = r'\b' + re.escape(phrase) + r'\b'
        cleaned = re.sub(pattern, '', cleaned)

    # eliminar espacios duplicados o basura
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()


    # ESTO SE PUEDE ACTIVAR O NO 
    #if len(cleaned) < 3: #si queda muy corto -> usar original
     #   return query

    return cleaned
#Inicializar estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

#CONTROLES EN SIDEBAR
st.sidebar.markdown("---")
#RAG
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

#intento de precarga para informar estado

#precarga para informar estado
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

#Mostrar mensajes anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


#CONFIGURACION DEL WEB TOOL
max_recent_web_context = 3  # número máximo de interacciones recientes para contexto web

#input del usuario
user_input = st.chat_input("Escribe tu pregunta...")

if user_input:
    #mostrar mensaje del usuario
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

    #buscar en la web si está activado
    if use_web:
        # construir query contextualizada -> si hay historial web reciente -> agregar contexto
        search_query = user_input
        if len(st.session_state.messages) >= max_recent_web_context:
            # tomar los últimos 2 mensajes del usuario para contexto
            recent_context = []
            for msg in st.session_state.messages[-(max_recent_web_context*2):]:  # ultimos mensajes (2 pares usuario-asistente)
                if msg["role"] == "user":
                    recent_context.append(msg["content"])

            if recent_context:
                # combinar contexto reciente con la pregunta actual
                search_query = f"{' '.join(recent_context[-max_recent_web_context:])} {user_input}" #ESTO HACE QUE SE DUPLIQUE LA ULTIMA PREGUNTA EN LA BUSQUEDA WEB, NO HAY MUCHO PROBLEMA

        #limpiar la query removiendo palabras innecesarias
        cleaned_query = clean_search_query(search_query)

        try:
            with st.spinner(f"Buscando: '{cleaned_query[:60]}...'"):
                results = search_web(cleaned_query, limit=web_limit)
        except RuntimeError as e:
            st.warning(str(e))
            results = []
        except Exception as e:
            st.error(f"Error buscando en la web: {e}")
            results = []

        # mostrar resultados
        if not results:
            st.info("No se encontraron resultados en la web para la consulta.")
        else:
            # mostrar el query limpio que se usó
            expander_title = f"Resultados de búsqueda: '{cleaned_query[:50]}...'"

            with st.expander(expander_title, expanded=True):
                for i, r in enumerate(results, start=1):
                    st.markdown(f"**{i}. {r.get('title', 'Sin título')}**")
                    if r.get('snippet'):
                        st.markdown(r.get('snippet'))
                    if r.get('url'):
                        st.markdown(f"[{r.get('url')}]({r.get('url')})")
                    st.markdown("---")

            #construir contexto para el modelo
            lines = []
            for i, r in enumerate(results, start=1):
                title = r.get('title', '')
                snippet = r.get('snippet', '')
                url = r.get('url', '')
                lines.append(f"{i}. {title}\n{snippet}\nFuente: {url}\n")
            web_context = "\n".join(lines)

    #construir mensajes del chat
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
        web_message = f"[Información de búsqueda web para tu consulta]\n\n{web_context}"
        st.session_state.messages.append({
            "role": "system",
            "content": web_message
        })

    #construir mensajes del chat
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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

                #mostrar fuentes RAG debajo 
                if rag_sources:
                    st.markdown("\n**Fuentes citadas (RAG):**")
                    for r in rag_sources:
                        st.markdown(f"- [{r['rank']}] {r['fuente']} — Autor: {r['autor']} — Semana: {r['semana']}")

    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")
