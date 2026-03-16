import streamlit as st
import sys
from pathlib import Path

import base64


st.set_page_config(
    page_title="Multi-Modal RAG Chatbot",
    layout="wide"
)

def load_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
logo_base64 = load_logo_base64("assets/logo2.png")


st.markdown(f"""
<div class="header-container">
    <img src="data:image/png;base64,{logo_base64}" class="app-logo">
    <div>
        <h1>Multi-Modal Document QA</h1>
        <p>Ask questions about documents using AI</p>
    </div>
</div>
""", unsafe_allow_html=True)




ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from QA.answer_generator import answer_question
st.markdown("""
<div class="upload-card">
<h3>📄 Upload Document</h3>
<p>Upload a PDF to start asking questions</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["pdf"],
    label_visibility="collapsed"
)
if uploaded_file:
    st.info(f"📄 Uploaded: **{uploaded_file.name}** ({round(uploaded_file.size/1024,1)} KB)")

# Detect new uploaded file
if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.doc_ready = False




#CSS
st.markdown("""
<style>

.stApp {
    background: #0C2C55;
    font-family: 'Inter', sans-serif;
}

/* Upload Card */

.upload-card{
    background: rgba(255,255,255,0.08);
    padding:25px;
    border-radius:16px;
    backdrop-filter: blur(15px);
    margin-bottom:20px;
}

            .header-container{
    display:flex;
    align-items:center;
    gap:15px;
}

.app-logo{
    width:60px;
    height:auto;
}
/* Chat container */

.stChatMessage{
    background: rgba(0,0,0,0.25);
    border-radius:14px;
    padding:14px;
    margin-bottom:12px;
    border:1px solid rgba(255,255,255,0.08);
}

/* User bubble */

.stChatMessage[data-testid="stChatMessage-user"]{
    background: linear-gradient(135deg,#2b2b3a,#1e1e2a);
    border-left:4px solid #7c5cff;
}

/* Assistant bubble */

.stChatMessage[data-testid="stChatMessage-assistant"]{
    background: rgba(255,255,255,0.08);
    border-left:4px solid #ffb347;
}

/* Chat input */

.stChatInput{
    border-radius:12px;
}

/* Upload drag area */

[data-testid="stFileUploader"]{
    background: rgba(255,255,255,0.08);
    border-radius:16px;
    padding:20px;
}

/* Success message */

[data-testid="stSuccess"]{
    background: rgba(0,255,150,0.15);
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

#For the title


st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! 👋 Ask me anything about the document."
        }
    ]


with st.sidebar:

    st.markdown("## 👤 About")

    with st.expander("Neil Parkhe - Maker"):
        st.markdown(
            """
            <style>
            .about-item {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
                font-size: 15px;
            }
            .about-item img {
                width: 20px;
                height: 20px;
            }
            .about-item a {
                color: #d8c7ff;
                text-decoration: none;
            }
            .about-item a:hover {
                text-shadow: 0 0 6px rgba(180,120,255,0.8);
            }
            </style>

            <div class="about-item">
                <img src="https://cdn-icons-png.flaticon.com/512/3524/3524659.png">
                <span>ML • Data Science • RAG Systems</span>
            </div>

            <div class="about-item">
                <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png">
                <a href="mailto:neilparkhe@gmail.com">neilparkhe@gmail.com</a>
            </div>

            <div class="about-item">
                <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png">
                <a href="https://github.com/Neil-05" target="_blank">GitHub</a>
            </div>

            <div class="about-item">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
                <a href="https://www.linkedin.com/in/neil-parkhe/" target="_blank">LinkedIn</a>
            </div>
            """,
            unsafe_allow_html=True
        )



    st.markdown("## ⚙️ System Info")
    st.markdown(
        """
        **Embedding:** all-MiniLM-L6-v2  
        **Vector DB:** FAISS  
        **LLM:** Groq  
        """
    )


    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Ask a new question!"
            }
        ]
        st.rerun()


from pathlib import Path

if uploaded_file is not None and not st.session_state.get("doc_ready", False):

    upload_dir = Path("data/raw_docs")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.doc_path = str(file_path)
    st.session_state.doc_ready = False


if "doc_path" in st.session_state and not st.session_state.doc_ready:

    progress_bar = st.progress(0)
    status_text = st.empty()
    with st.spinner("Extracting document content..."):

        from Ingestion.text_parse import texts as extract_text
        from Ingestion.text_parse import images as extract_images_and_ocr
        from Ingestion.text_parse import tables as extract_tables
        from Chunking.chunker import chunk_content
        from Vector_store.build_index import build_embeddings

        import json

        pdf_path = st.session_state.doc_path

        # Ensure folders exist
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("data/embeddings").mkdir(parents=True, exist_ok=True)
        status_text.markdown("🔍 **Step 1/4: Extracting text, images, tables... (20%)**")
        progress_bar.progress(20)
        # Extract modalities
        text_data = extract_text(pdf_path)
        image_data = extract_images_and_ocr(pdf_path)
        table_data = extract_tables(pdf_path)
        status_text.markdown("🔍 **Step 2/4: Processing extracted data... (50%)**")
        progress_bar.progress(50)

        # Save extracted JSON
        with open("data/processed/text.json", "w") as f:
            json.dump(text_data, f, indent=2)

        with open("data/processed/images.json", "w") as f:
            json.dump(image_data, f, indent=2)

        with open("data/processed/tables.json", "w") as f:
            json.dump(table_data, f, indent=2)

        # Merge JSON files
        all_data = text_data + image_data + table_data

        with open("data/processed/all_data.json", "w") as f:
            json.dump(all_data, f, indent=2)

        # Chunk
        chunk_content(
            "data/processed/all_data.json",
            "data/processed/chunks.json"
        )
        status_text.markdown("✂️ **Step 3/4: Chunking document... (80%)**")
        progress_bar.progress(80)

        # Build embeddings
        build_embeddings(
             Path("data/processed/chunks.json"),
             Path("data/embeddings/vector.index"),
            Path("data/embeddings/metadata.pkl")
        )
        status_text.markdown("🧠 **Step 4/4: Building embeddings... (100%)**")
        progress_bar.progress(100)

    st.session_state.doc_ready = True
    st.success("Document processed. You can now ask questions!")



for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "citations" in msg:
            with st.expander("📚 Sources"):
                for c in msg["citations"]:
                    st.markdown(f"- {c}")





prompt = st.chat_input("Ask a question about the document...")

st.markdown("### 💡 Example questions")

col1,col2,col3 = st.columns(3)

if col1.button("📑 Summarize document"):
    st.session_state.example="Summarize the document"

if col2.button("📊 Extract key insights"):
    st.session_state.example="What are the key insights?"

if col3.button("📌 Important facts"):
    st.session_state.example="List important facts"

if prompt:
    # user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

   #assistant replying
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            answer, citations = answer_question(prompt)

        st.markdown(answer)

        with st.expander("📚 Sources"):
            for c in citations:
                st.markdown(f"- {c}")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })
