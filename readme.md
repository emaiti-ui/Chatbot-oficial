# README.md — Chatbot RAG para Ingeniería Bioquímica

Este proyecto implementa un chatbot basado en Retrieval-Augmented Generation (RAG) para asistir al área de investigación de la carrera de Ingeniería Bioquímica, usando documentos cargados localmente (como PDFs) y el modelo GPT de OpenAI. El chatbot responde preguntas específicas buscando primero en los documentos cargados y luego generando respuestas fundamentadas.

---

## Funcionalidades
 Busca respuestas en documentos PDF.  
 Usa modelos de lenguaje avanzados (GPT-4, GPT-3.5-turbo).  
 Guarda índice vectorial localmente con FAISS para optimizar búsquedas.  
 Interfaz web amigable con Streamlit.  
 Preparado para correr localmente o desplegar en Streamlit Cloud.

---

##  Estructura del proyecto

```
chatbot_bioq/
├── .env                     # Claves locales (no subir a GitHub)
├── .gitignore               # Ignora archivos sensibles
├── .streamlit/
│   └── secrets.toml         # Claves para Streamlit Cloud
├── docs/
│   └── tus_archivos.pdf     # Documentos fuente
├── faiss_index/             # Índice generado (se crea automáticamente)
├── requirements.txt         # Dependencias del proyecto
├── app.py                   # Código principal
└── README.md                # Este archivo
```

---

##  Instalación

1 Clona el repositorio:
```
git clone https://github.com/tu-usuario/chatbot_bioq.git
cd chatbot_bioq
```

2 Crea entorno virtual:
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3 Instala dependencias:
```
pip install -r requirements.txt
```

4 Configura tus claves:
- **Local**: crea `.env` con tu clave `OPENAI_API_KEY`.
- **Streamlit Cloud**: usa `.streamlit/secrets.toml`.

5 Coloca tus PDFs en `docs/`.

6 Ejecuta la app:
```
streamlit run app.py
```

Abre tu navegador en `http://localhost:8501`.

---

##  Cómo contribuir
 Haz un fork del proyecto.  
 Crea una rama con tu mejora (`git checkout -b mejora-x`).  
 Realiza tus cambios.  
 Haz commit (`git commit -m "Agrega mejora x"`).  
 Haz push a tu rama (`git push origin mejora-x`).  
 Abre un pull request.

---

##  Notas
- Cambia de modelo (`gpt-4` → `gpt-3.5-turbo`) si necesitas reducir costos.
- Borra la carpeta `faiss_index/` si cambias los documentos para regenerar el índice.
- Este proyecto está diseñado como base: puedes extenderlo para incluir otros formatos de documentos, conexiones a bases de datos, o incluso integración con Pinecone para almacenar índices en la nube.

---

##  Licencia
MIT License — libre para uso académico y personal.