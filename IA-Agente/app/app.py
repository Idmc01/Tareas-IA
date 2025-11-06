import os
import streamlit as st
from openai import OpenAI
from tools.web_search import search_web

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#----- Función para limpiar query de búsqueda
def clean_search_query(query: str) -> str:
    # Limpia la query removiendo palabras comunes que no aportan a la búsqueda
    stopwords = [
        'dame', 'dime', 'que es', 'qué es', 'cual es', 'cuál es', 'como es', 'cómo es',
        'cuales son', 'cuáles son', 'como son', 'cómo son', 'me puedes', 'puedes',
        'explicame', 'explícame', 'explica', 'dime sobre', 'dame información',
        'quiero saber', 'necesito saber', 'me gustaría', 'quisiera',
        'por favor', 'gracias', 'hola', 'ayudame', 'ayúdame'
    ]
    
    query_lower = query.lower()
    cleaned = query
    
    #remove stopwords
    for word in stopwords:
        #buscar al inicio de la frase
        if query_lower.startswith(word + ' '):
            cleaned = query[len(word):].strip()
            query_lower = cleaned.lower()
        #buscar en cualquier parte con espacios alrededor
        cleaned = ' '.join([w for w in cleaned.split() if w.lower() not in word.split()])
    
    # ESTO SE PUEDE ACTIVAR O NO 
    #if len(cleaned.strip()) < 3: #si queda muy corto -> usar original
     #   return query
    
    return cleaned.strip()

#----- Inicializar estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

#----- MOVER LOS CONTROLES A LA SIDEBAR
st.sidebar.markdown("---")
st.sidebar.subheader("Configuración de Búsqueda Web")
use_web = st.sidebar.checkbox("Buscar en la web antes de responder", value=False)
web_limit = st.sidebar.slider("Número de resultados web", min_value=1, max_value=10, value=3)

#----- Mostrar mensajes anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#CONFIGURACION DEL WEB TOOL
max_recent_web_context = 3  # número máximo de interacciones recientes para contexto web

#----- Input del usuario
user_input = st.chat_input("Escribe tu pregunta...")

if user_input:
    # Mostrar mensaje del usuario
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    web_context = None
    
    # Buscar en la web si está activado
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
        
        # Limpiar la query removiendo palabras innecesarias
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

            # Construir contexto para el modelo
            lines = []
            for i, r in enumerate(results, start=1):
                title = r.get('title', '')
                snippet = r.get('snippet', '')
                url = r.get('url', '')
                lines.append(f"{i}. {title}\n{snippet}\nFuente: {url}\n")
            web_context = "\n".join(lines)

    # Si hay contexto web, agregarlo al historial para que persista
    if web_context:
        web_message = f"[Información de búsqueda web para tu consulta]\n\n{web_context}"
        st.session_state.messages.append({
            "role": "system",
            "content": web_message
        })
    
    # Construir mensajes del chat
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

    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")