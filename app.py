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

# ================== CONFIGURACIÓN DE PÁGINA MEJORADA ==================
st.set_page_config(
    page_title="BioChatBot 🧪",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# BioChatBot 🧬\n¡Tu asistente de investigación en Ingeniería Bioquímica!"
    }
)

# ================== CSS PERSONALIZADO CORREGIDO ==================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .sidebar-header {
        background: linear-gradient(45deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .info-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #333333 !important;
    }
    
    .info-card h4 {
        color: #667eea !important;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        color: #444444 !important;
        line-height: 1.6;
        margin-bottom: 0;
    }
    
    .stTextInput > div > div > input {
        background: linear-gradient(90deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #e9ecef;
        border-radius: 25px;
        padding: 12px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
        color: #333333 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        color: #333333 !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #888888 !important;
    }
    
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem 1rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        transition: all 0.3s ease;
        margin: 1rem 0;
        color: #333333 !important;
    }
    
    .upload-zone:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #667eea33 0%, #764ba233 100%);
    }
    
    .success-message {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .warning-message {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    /* Estilos específicos para elementos de Streamlit */
    .stMarkdown p {
        color: #333333 !important;
    }
    
    .stExpander .streamlit-expanderHeader {
        color: #333333 !important;
    }
    
    .stExpander .streamlit-expanderContent {
        color: #333333 !important;
    }
    
    /* Asegurar contraste en texto general */
    div[data-testid="stMarkdownContainer"] p {
        color: #333333 !important;
    }
    
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {
        color: #333333 !important;
    }
    
    /* Corregir color en métricas y otros elementos */
    .metric-container {
        color: #333333 !important;
    }
    
    /* Sidebar específico */
    .stSidebar .stMarkdown p {
        color: #333333 !important;
    }
    
    .stSidebar .stMarkdown h1,
    .stSidebar .stMarkdown h2,
    .stSidebar .stMarkdown h3,
    .stSidebar .stMarkdown h4 {
        color: #333333 !important;
    }
    
    .stSidebar .stMarkdown li {
        color: #333333 !important;
    }
    
    .stSidebar .stMarkdown strong {
        color: #333333 !important;
    }
    
    /* Métricas en sidebar */
    .stSidebar .stMetric {
        color: #333333 !important;
    }
    
    .stSidebar .stMetric > div {
        color: #333333 !important;
    }
    
    /* Expandibles en sidebar */
    .stSidebar .stExpander {
        color: #333333 !important;
    }
    
    .stSidebar .stExpander summary {
        color: #333333 !important;
    }
    
    .stSidebar .stExpander > div {
        color: #333333 !important;
    }
    
    /* Footer y contenido principal */
    .stApp > div {
        color: #333333 !important;
    }
    
    /* Texto general de la aplicación */
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #333333 !important;
    }
    
    /* Estilos para el historial de chat */
    .chat-question {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        color: #1565c0 !important;
    }
    
    .chat-answer {
        background: #f3e5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #9c27b0;
        margin: 0.5rem 0;
        color: #7b1fa2 !important;
    }
</style>
""", unsafe_allow_html=True)

# ================== HEADER PRINCIPAL CON ESTILO ==================
st.markdown("""
<div class="main-header">
    <h1>🧬 BioChatBot Avanzado</h1>
    <p>Tu Asistente Inteligente de Investigación en Ingeniería Bioquímica</p>
</div>
""", unsafe_allow_html=True)

# ================== SIDEBAR ESTÉTICO MEJORADO ==================
with st.sidebar:
    st.markdown('<div class="sidebar-header">📁 Gestión de Documentos</div>', unsafe_allow_html=True)
    
    # Zona de carga con estilo mejorado
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    st.markdown("### 🎯 Sube tus documentos")
    uploaded_files = st.file_uploader(
        "Arrastra y suelta tus PDFs aquí",
        type=['pdf'],
        accept_multiple_files=True,
        help="💡 Puedes subir múltiples archivos PDF simultáneamente",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Opción para usar carpeta local con estilo
    use_local_folder = st.checkbox("📂 Usar carpeta local ./docs")
    
    if use_local_folder:
        st.markdown("""
        <div class="info-card">
            <strong style="color: #333333;">📋 Modo:</strong> <span style="color: #FFFFFF;">Carpeta Local</span><br>
            <strong style="color: #333333;">📁 Ruta:</strong> <span style="color: #FFFFFF;">./docs</span>
        </div>
        """, unsafe_allow_html=True)

# Inicializar variables
documents = []
temp_dir = None

# ================== CARGA DE DOCUMENTOS CON MENSAJES ESTÉTICOS ==================
if uploaded_files:
    # Procesar archivos subidos
    temp_dir = tempfile.mkdtemp()
    
    with st.spinner("🔄 Procesando archivos subidos..."):
        for uploaded_file in uploaded_files:
            # Guardar archivo temporalmente
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Cargar documentos desde carpeta temporal
        pdf_loader = DirectoryLoader(temp_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = pdf_loader.load()
    
    # Mensaje de éxito con animación
    st.balloons()
    st.markdown(f"""
    <div class="success-message">
        🎉 ¡Éxito! {len(uploaded_files)} archivo(s) cargado(s) correctamente
    </div>
    """, unsafe_allow_html=True)

elif use_local_folder:
    # Cargar documentos de la carpeta ./docs
    if os.path.exists("./docs"):
        with st.spinner("📂 Cargando documentos de ./docs..."):
            pdf_loader = DirectoryLoader("./docs", glob="*.pdf", loader_cls=PyPDFLoader)
            documents = pdf_loader.load()
        
        if documents:
            st.markdown(f"""
            <div class="success-message">
                ✅ {len(documents)} documento(s) cargado(s) desde ./docs
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-message">
                ⚠️ No se encontraron documentos PDF en la carpeta ./docs
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-message">
            ❌ La carpeta ./docs no existe
        </div>
        """, unsafe_allow_html=True)
        
else:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
        <h3>🚀 ¡Bienvenido al BioChatBot!</h3>
        <p>👆 Por favor, sube documentos PDF o selecciona usar la carpeta local para comenzar</p>
        <p>💡 Una vez cargados, podrás hacer preguntas inteligentes sobre tu contenido</p>
    </div>
    """, unsafe_allow_html=True)

# Verificar que se cargaron documentos
if not documents:
    st.stop()

# ================== PROCESAMIENTO DE DOCUMENTOS ==================
# Paso 2: Dividir en chunks
with st.spinner("✂️ Dividiendo documentos en fragmentos..."):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

# Paso 3: Crear embeddings con Gemini
with st.spinner("🔍 Generando embeddings con Gemini..."):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Paso 4: Crear o cargar vectorstore
vectorstore_key = "default" if use_local_folder else "uploaded"
faiss_index_path = f"faiss_index_{vectorstore_key}"

if os.path.exists(faiss_index_path) and use_local_folder:
    try:
        with st.spinner("📋 Cargando índice FAISS existente..."):
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        
        st.markdown("""
        <div class="success-message">
            ⚡ Índice FAISS cargado desde caché - ¡Súper rápido!
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"⚠️ Error al cargar índice: {e}")
        with st.spinner("🏗️ Creando nuevo índice FAISS..."):
            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local(faiss_index_path)
else:
    with st.spinner("🏗️ Generando índice vectorial FAISS..."):
        vectorstore = FAISS.from_documents(texts, embeddings)
        if use_local_folder:  # Solo guardar índice si usa carpeta local
            vectorstore.save_local(faiss_index_path)
    
    st.markdown("""
    <div class="success-message">
        🎯 Índice FAISS creado exitosamente - ¡Listo para consultas!
    </div>
    """, unsafe_allow_html=True)

# ================== CONFIGURACIÓN DEL MODELO GEMINI ==================
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

# ================== HISTORIAL DE CHAT MEJORADO ==================
if "history" not in st.session_state:
    st.session_state.history = []

# ================== ENTRADA DE TEXTO CON ESTILO ==================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown("### 💬 Haz tu pregunta")
query = st.text_input(
    "",
    placeholder="💭 Escribe tu pregunta sobre investigación bioquímica aquí...",
    key="input",
    help="🔍 Pregunta cualquier cosa sobre tus documentos cargados"
)
st.markdown('</div>', unsafe_allow_html=True)

# ================== PROCESAMIENTO DE CONSULTAS ==================
if query:
    with st.spinner("🤖 Generando respuesta inteligente..."):
        try:
            result = qa_chain.run(query)
            st.session_state.history.append((query, result))
            
            # Mensaje de éxito sutil
            st.success("✨ Respuesta generada exitosamente")
            
        except Exception as e:
            st.markdown(f"""
            <div class="warning-message">
                ❌ Oops! Algo salió mal: {str(e)}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("💡 **Sugerencia:** Verifica tu conexión y los documentos cargados.")

# ================== HISTORIAL VISUAL CORREGIDO ==================
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 💭 Historial de Conversación")
    
    for i, (q, r) in enumerate(reversed(st.session_state.history)):
        with st.expander(f"🗨️ {q[:60]}..." if len(q) > 60 else f"🗨️ {q}", expanded=(i==0)):
            # Pregunta con estilo mejorado
            st.markdown(f"""
            <div class="chat-question">
                <h4 style="color: #1565c0; margin-bottom: 10px;">❓ Pregunta:</h4>
                <p style="font-size: 16px; line-height: 1.6; color: #333333; margin: 0;">{q}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Respuesta con estilo mejorado
            st.markdown(f"""
            <div class="chat-answer">
                <h4 style="color: #7b1fa2; margin-bottom: 10px;">🤖 Respuesta:</h4>
                <p style="font-size: 16px; line-height: 1.6; color: #333333; margin: 0;">{r}</p>
            </div>
            """, unsafe_allow_html=True)

# ================== SIDEBAR CON INFORMACIÓN ESTÉTICA ==================
with st.sidebar:
    st.markdown("---")
    st.markdown('<div class="sidebar-header">📊 Estadísticas del Sistema</div>', unsafe_allow_html=True)
    
    if documents:
        # Métricas visuales
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Documentos", len(documents), delta="Cargados")
        with col2:
            st.metric("📝 Chunks", len(texts), delta="Procesados")
        
        # Barra de progreso
        progress = min(len(documents) / 10, 1.0)  # Máximo 10 docs como 100%
        st.progress(progress, text=f"💾 Capacidad utilizada: {int(progress*100)}%")
        
        # Mostrar nombres de archivos en formato estético
        file_names = list(set([doc.metadata.get('source', 'Desconocido').split('/')[-1] for doc in documents]))
        with st.expander("📋 Archivos cargados", expanded=False):
            for name in file_names[:5]:  # Mostrar máximo 5
                st.markdown(f"<span style='color: #FFFFFF;'>• **{name}**</span>", unsafe_allow_html=True)
            if len(file_names) > 5:
                st.markdown(f"<span style='color: #FFFFFF;'>... y {len(file_names)-5} más</span>", unsafe_allow_html=True)
        
        # Información del sistema en card estético
        st.markdown(f"""
        <div class="info-card">
            <strong style="color: #333333;">🤖 Modelo:</strong> <span style="color: #FFFFFF;">Gemini 1.5 Flash</span><br>
            <strong style="color: #333333;">🔍 Embeddings:</strong> <span style="color: #FFFFFF;">models/embedding-001</span><br>
            <strong style="color: #333333;">⚡ Estado:</strong> <span style="color: #FFFFFF;">●</span> <span style="color: #FFFFFF;">Activo</span><br>
            <strong style="color: #333333;">🔄 Última actualización:</strong> <span style="color: #FFFFFF;">Ahora</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Botón para limpiar historial
    if st.button("🗑️ Limpiar Historial", type="secondary"):
        st.session_state.history = []
        st.experimental_rerun()

# ================== FOOTER PROFESIONAL ==================
st.markdown("---")
st.markdown("""
<div style="text-align: center; 
            padding: 2rem; 
            background: linear-gradient(90deg, #667eea22 0%, #764ba222 100%); 
            border-radius: 15px; 
            margin-top: 2rem;
            color: #333333;">
    <h4 style="color: #FFFFFF;">🎓 Desarrollado para Prácticas Profesionales</h4>
    <p style="color: #FFFFFF; font-size: 18px;">Ingeniería Bioquímica • Powered by Gemini AI</p>
    <p style="font-size: 14px; color: #FFFFFF;">© 2025 • Hecho con ❤️ y mucho ☕</p>
    <div style="margin-top: 1rem;">
        <span style="margin: 0 10px;">🧬</span>
        <span style="margin: 0 10px;">⚗️</span>
        <span style="margin: 0 10px;">🔬</span>
        <span style="margin: 0 10px;">📊</span>
        <span style="margin: 0 10px;">🤖</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== LIMPIEZA DE ARCHIVOS TEMPORALES ==================
if temp_dir and os.path.exists(temp_dir):
    import atexit
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))