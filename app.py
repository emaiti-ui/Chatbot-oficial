import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
import shutil

# Cargar variables de entorno desde .env
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("GOOGLE_API_KEY no encontrado. Verifica tu archivo .env o secrets.toml.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

# Streamlit app setup
st.set_page_config(page_title="Chatbot Bioquímica", page_icon="🧪")
st.title("Chatbot de Investigación en Ingeniería Bioquímica")

# Sidebar para carga de documentos
with st.sidebar:
    st.header("📁 Cargar Documentos")
    
    # Opción 1: Subir archivos
    uploaded_files = st.file_uploader(
        "Sube tus documentos PDF:",
        type=['pdf'],
        accept_multiple_files=True,
        help="Puedes subir múltiples archivos PDF"
    )
    
    st.markdown("---")
    
    # Opción 2: Seleccionar carpeta local
    use_local_folder = st.checkbox("Usar carpeta local ./docs")
    
    if use_local_folder:
        st.info("Usando documentos de la carpeta ./docs")

# Inicializar variables
documents = []
temp_dir = None

# Paso 1: Cargar documentos según la opción seleccionada
if uploaded_files:
    # Procesar archivos subidos
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    with st.spinner("Procesando archivos subidos..."):
        for uploaded_file in uploaded_files:
            # Guardar archivo temporalmente
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Cargar documentos desde carpeta temporal
        pdf_loader = DirectoryLoader(temp_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = pdf_loader.load()
        
        st.success(f"✅ {len(uploaded_files)} archivo(s) cargado(s) exitosamente")

elif use_local_folder:
    # Cargar documentos de la carpeta ./docs
    if os.path.exists("./docs"):
        with st.spinner("Cargando documentos de ./docs..."):
            pdf_loader = DirectoryLoader("./docs", glob="*.pdf", loader_cls=PyPDFLoader)
            documents = pdf_loader.load()
        
        if documents:
            st.success(f"✅ {len(documents)} documento(s) cargado(s) desde ./docs")
        else:
            st.warning("No se encontraron documentos PDF en la carpeta ./docs")
    else:
        st.error("La carpeta ./docs no existe")

else:
    st.info("👆 Por favor, sube documentos o selecciona usar la carpeta local")

# Verificar que se cargaron documentos
if not documents:
    st.warning("⚠️ No hay documentos cargados. Por favor, sube archivos o usa la carpeta local.")
    st.stop()

# Paso 2: Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Paso 3: Crear embeddings con Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Paso 4: Crear o cargar vectorstore
vectorstore_key = "default" if use_local_folder else "uploaded"
faiss_index_path = f"faiss_index_{vectorstore_key}"

if os.path.exists(faiss_index_path) and use_local_folder:
    try:
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        st.success("✅ Índice FAISS cargado exitosamente")
    except Exception as e:
        st.warning(f"⚠️ Error al cargar índice existente: {e}. Creando nuevo índice...")
        with st.spinner("Generando índice vectorial FAISS..."):
            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local(faiss_index_path)
else:
    with st.spinner("Generando índice vectorial FAISS..."):
        vectorstore = FAISS.from_documents(texts, embeddings)
        if use_local_folder:  # Solo guardar índice si usa carpeta local
            vectorstore.save_local(faiss_index_path)
        st.success("✅ Índice FAISS creado exitosamente")

# Paso 5: Crear RAG agent con Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), 
    chain_type="stuff"
)

# Historial de chat
if "history" not in st.session_state:
    st.session_state.history = []

# Entrada del usuario
query = st.text_input("Escribe tu pregunta sobre investigación:", key="input")

if query:
    with st.spinner("Generando respuesta..."):
        try:
            result = qa_chain.run(query)
            st.session_state.history.append((query, result))
        except Exception as e:
            st.error(f"Error al generar respuesta: {e}")

# Mostrar historial
if st.session_state.history:
    st.subheader("📜 Historial")
    for q, r in reversed(st.session_state.history):
        with st.expander(f"Pregunta: {q[:50]}..."):
            st.write(f"**Pregunta:** {q}")
            st.write(f"**Respuesta:** {r}")
            st.markdown("---")

# Sidebar con información
with st.sidebar:
    st.markdown("---")
    st.header("ℹ️ Información del Sistema")
    
    if documents:
        st.write(f"📊 Documentos cargados: {len(documents)}")
        st.write(f"📝 Chunks de texto: {len(texts)}")
        
        # Mostrar nombres de archivos
        file_names = list(set([doc.metadata.get('source', 'Desconocido').split('/')[-1] for doc in documents]))
        with st.expander("📋 Archivos cargados"):
            for name in file_names:
                st.write(f"• {name}")
    
    st.write("🤖 Modelo: Gemini 1.5 Flash")
    st.write("🔍 Embeddings: models/embedding-001")

# Limpiar archivos temporales al final
if temp_dir and os.path.exists(temp_dir):
    import atexit
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

st.markdown("---")
st.markdown("_Desarrollado para prácticas profesionales de Ingeniería Bioquímica_")