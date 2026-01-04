import streamlit as st
import sys
from pathlib import Path

import base64

def load_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
logo_base64 = load_logo_base64("assets/logo2.png")



ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from QA.answer_generator import answer_question


st.set_page_config(
    page_title="Multi-Modal RAG Chatbot",
    layout="wide"
)

#CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #E48AED, #7b3fe4);
        color: white;
    }



   .glow-title {
    font-size: 34px;
    font-weight: 700;
    color: #f3e8ff; /* soft lavender text */
    text-shadow:
        0 0 6px rgba(180,120,255,0.6),
        0 0 14px rgba(140,180,255,0.6),
        0 0 28px rgba(120,90,255,0.8);
    animation: glowPulse 3.5s ease-in-out infinite;
}

@keyframes glowPulse {
    0% {
        text-shadow:
            0 0 6px rgba(180,120,255,0.5),
            0 0 14px rgba(140,180,255,0.5),
            0 0 26px rgba(120,90,255,0.6);
    }
    50% {
        text-shadow:
            0 0 12px rgba(200,150,255,0.9),
            0 0 26px rgba(160,210,255,0.9),
            0 0 40px rgba(150,120,255,1);
    }
    100% {
        text-shadow:
            0 0 6px rgba(180,120,255,0.5),
            0 0 14px rgba(140,180,255,0.5),
            0 0 26px rgba(120,90,255,0.6);
    }
}


.glow-logo {
    width: 55px;
    filter: drop-shadow(0 0 6px rgba(255,255,255,0.6))
            drop-shadow(0 0 14px rgba(180,120,255,0.7))
            drop-shadow(0 0 26px rgba(140,80,255,0.8));
    animation: logoFloatGlow 3s ease-in-out infinite;
}

@keyframes logoFloatGlow {
    0% {
        transform: translateY(0px) scale(1);
        filter: drop-shadow(0 0 6px rgba(255,255,255,0.5))
                drop-shadow(0 0 14px rgba(180,120,255,0.6))
                drop-shadow(0 0 26px rgba(140,80,255,0.7));
    }
    50% {
        transform: translateY(-4px) scale(1.03);
        filter: drop-shadow(0 0 10px rgba(255,255,255,0.9))
                drop-shadow(0 0 22px rgba(200,150,255,0.9))
                drop-shadow(0 0 36px rgba(170,120,255,1));
    }
    100% {
        transform: translateY(0px) scale(1);
        filter: drop-shadow(0 0 6px rgba(255,255,255,0.5))
                drop-shadow(0 0 14px rgba(180,120,255,0.6))
                drop-shadow(0 0 26px rgba(140,80,255,0.7));
    }
}




    label, p, h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    .stChatMessage {
        background-color: rgba(0,0,0,0.25);
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
    background-color: #1f1f2e;
}

.stChatMessage[data-testid="stChatMessage-assistant"] {
    background-color: rgba(0,0,0,0.35);
}


    .stSpinner > div {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#For the title
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:14px; height:70px;">
        <img src="data:image/png;base64,{logo_base64}" class="glow-logo"/>
        <h2 class="glow-title" style="margin:0;">
            Multi-Modal Document QA
        </h2></div>
    
    """,
    unsafe_allow_html=True
)


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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "citations" in msg:
            with st.expander("📚 Sources"):
                for c in msg["citations"]:
                    st.markdown(f"- {c}")





prompt = st.chat_input("Ask a question about the document...")

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
