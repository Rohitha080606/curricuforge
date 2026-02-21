import streamlit as st
import numpy as np
import os
import fitz
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()


# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

client = InferenceClient(token=HF_API_KEY)


def generate_curriculum(topic, academic_level, duration, program_focus, evaluation_framework):

    prompt = f"""
    Create a structured {duration}-week university curriculum.

    Course Title: {topic}
    Academic Level: {academic_level}
    Program Focus: {program_focus}
    Evaluation Framework: {evaluation_framework}

    STRICT REQUIREMENTS:

    1. The course MUST be exactly {duration} weeks.
    2. Provide a weekly breakdown from Week 1 to Week {duration}.
    3. Do NOT exceed or reduce the number of weeks.
    4. Each week must contain:
       - Topic
       - Key Concepts
       - Practical / Activity Component
    5. After weekly plan include:
       - Learning Outcomes (max 4 bullet points)
       - Assessment Strategy aligned to {evaluation_framework}
       - Relevant Job Roles (max 3 roles)
    6. No introduction paragraph.
    7. No repetition.
    8. Follow clean structured format.

    Format:

    Course Name:
    Level:
    Duration: {duration} Weeks

    Weekly Plan:
    Week 1:
    Week 2:
    ...
    Week {duration}:

    Learning Outcomes:
    -
    -

    Assessment Strategy:
    -

    Relevant Job Roles:
    -
    """

    response = client.chat_completion(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1800,
        temperature=0.3 
    )

    return response.choices[0].message.content
    
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem
from reportlab.lib.pagesizes import A4
from io import BytesIO

import re

def clean_markdown(text):
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)
    return text

def generate_pdf(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    heading_style = styles["Heading1"]

    elements = []

    for line in text.split("\n"):
        if line.strip() == "":
            elements.append(Spacer(1, 0.2 * inch))
        elif "Course" in line or "Week" in line:
            elements.append(Paragraph(line, heading_style))
            elements.append(Spacer(1, 0.2 * inch))
        else:
            elements.append(Paragraph(line, normal_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Streamlit UI
st.set_page_config(page_title="CurricuForge MVP")
st.set_page_config(
    page_title="CurricuForge",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Fira+Code:wght@400&display=swap');
*,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;-webkit-font-smoothing:antialiased}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2.5rem 3.5rem!important;max-width:980px!important}

/* BG */
.stApp{background:#0a2540;background-image:radial-gradient(ellipse 80% 50% at 50% -15%,rgba(99,179,237,.1) 0%,transparent 60%),radial-gradient(ellipse 40% 30% at 90% 70%,rgba(99,91,255,.07) 0%,transparent 55%)}

/* SIDEBAR */
[data-testid="stSidebar"]{background:#071a30!important;border-right:1px solid rgba(255,255,255,.06)!important}
[data-testid="stSidebar"] *{color:#c8d8ea!important}
[data-testid="stSidebar"] label{font-size:.63rem!important;font-weight:600!important;letter-spacing:.05em!important;text-transform:uppercase!important;color:rgba(99,179,237,.45)!important}
[data-testid="stSidebar"] .stSelectbox>div>div{background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.09)!important;border-radius:8px!important;transition:border-color .15s!important}
[data-testid="stSidebar"] .stSelectbox>div>div:hover{border-color:rgba(99,91,255,.45)!important}
[data-testid="stSidebar"] .stRadio>div>label{background:rgba(255,255,255,.03)!important;border:1px solid rgba(255,255,255,.08)!important;border-radius:7px!important;padding:.45rem .9rem!important;font-size:.8rem!important;transition:all .15s!important}
[data-testid="stSidebar"] .stRadio>div>label:hover{background:rgba(99,91,255,.07)!important;border-color:rgba(99,91,255,.4)!important;color:#e8f4fd!important}

/* INPUTS */
.stTextInput input{background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:8px!important;color:#e8f4fd!important;padding:.72rem 1rem!important;transition:all .15s!important}
.stTextInput input:focus{border-color:#635bff!important;background:rgba(99,91,255,.05)!important;box-shadow:0 0 0 3px rgba(99,91,255,.12)!important}
.stTextInput input::placeholder{color:rgba(200,216,234,.22)!important}
.stTextInput label{font-size:.63rem!important;font-weight:600!important;letter-spacing:.05em!important;text-transform:uppercase!important;color:rgba(99,179,237,.45)!important}

/* SELECT + RADIO */
div[data-baseweb="select"]>div{background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:8px!important;color:#e8f4fd!important;transition:border-color .15s!important}
div[data-baseweb="select"]>div:hover{border-color:rgba(99,91,255,.45)!important}
div[role="radiogroup"]>label{background:rgba(255,255,255,.03)!important;border:1px solid rgba(255,255,255,.09)!important;border-radius:7px!important;padding:7px 14px!important;color:rgba(200,216,234,.6)!important;font-size:.83rem!important;transition:all .15s!important}
div[role="radiogroup"]>label:hover{border-color:rgba(99,91,255,.4)!important;color:#e8f4fd!important}

/* FILE UPLOADER */
[data-testid="stFileUploader"]{background:rgba(255,255,255,.03)!important;border:1.5px dashed rgba(255,255,255,.1)!important;border-radius:10px!important;transition:all .15s!important}
[data-testid="stFileUploader"]:hover{border-color:rgba(99,91,255,.4)!important;background:rgba(99,91,255,.04)!important}
[data-testid="stFileUploader"] label{font-size:.63rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.05em!important;color:rgba(99,179,237,.4)!important}

/* BUTTONS */
.stButton>button{background:#635bff!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-size:.88rem!important;width:100%!important;padding:.72rem!important;transition:all .15s!important;box-shadow:0 2px 14px rgba(99,91,255,.4),inset 0 1px 0 rgba(255,255,255,.12)!important}
.stButton>button:hover{background:#7a73ff!important;transform:translateY(-1px)!important;box-shadow:0 6px 22px rgba(99,91,255,.45)!important}
.stButton>button:active{transform:translateY(0)!important;background:#4f46e5!important}
.stDownloadButton>button{background:rgba(255,255,255,.04)!important;color:rgba(200,216,234,.65)!important;border:1px solid rgba(255,255,255,.09)!important;border-radius:8px!important;font-size:.82rem!important;width:100%!important;padding:.65rem!important;transition:all .15s!important}
.stDownloadButton>button:hover{border-color:rgba(99,91,255,.4)!important;color:#e8f4fd!important;background:rgba(99,91,255,.07)!important}

/* MISC */
.stSuccess{background:rgba(16,185,129,.07)!important;border:1px solid rgba(16,185,129,.22)!important;border-radius:8px!important}
.stWarning{background:rgba(245,158,11,.07)!important;border:1px solid rgba(245,158,11,.22)!important;border-radius:8px!important}
.stSpinner>div{border-top-color:#635bff!important}
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-thumb{background:rgba(99,91,255,.2);border-radius:10px}
::-webkit-scrollbar-thumb:hover{background:rgba(99,91,255,.4)}
</style>

<div style="border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:2rem;margin-bottom:2.5rem;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:2.5rem;">
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="width:28px;height:28px;background:linear-gradient(135deg,#635bff,#0ea5e9);border-radius:8px;display:grid;place-items:center;font-size:.8rem;box-shadow:0 2px 10px rgba(99,91,255,.45);">⚡</div>
      <span style="font-weight:700;font-size:1rem;color:#e8f4fd;letter-spacing:-.01em;">CurricuForge</span>
      <span style="font-family:'Fira Code',monospace;font-size:.58rem;background:rgba(99,91,255,.12);border:1px solid rgba(99,91,255,.28);color:#a5b4fc;padding:2px 9px;border-radius:4px;">beta</span>
    </div>
    <span style="font-family:'Fira Code',monospace;font-size:.6rem;color:rgba(200,216,234,.28);letter-spacing:.05em;">Mistral 7B · RAG · 60+ courses</span>
  </div>

  <div style="font-weight:700;font-size:2.2rem;letter-spacing:-.03em;line-height:1.18;color:#e8f4fd;margin-bottom:.8rem;">
    Build better curricula,<br>
    <span style="background:linear-gradient(90deg,#63b3ed,#635bff,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">powered by AI.</span>
  </div>
  <div style="font-size:.88rem;color:rgba(200,216,234,.45);line-height:1.65;max-width:500px;margin-bottom:2rem;">
    Generate structured academic curricula in seconds. Backed by Mistral 7B with RAG support for document-based generation.
  </div>

  <div style="display:flex;gap:.75rem;flex-wrap:wrap;">
    <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:.9rem 1.4rem;display:flex;align-items:center;gap:.65rem;">
      <div style="width:7px;height:7px;border-radius:50%;background:#635bff;box-shadow:0 0 6px rgba(99,91,255,.6);"></div>
      <div><div style="font-weight:600;font-size:.78rem;color:#e8f4fd;">Wide Curriculum Coverage</div><div style="font-family:'Fira Code',monospace;font-size:.58rem;color:rgba(200,216,234,.3);margin-top:1px;">Database indexed</div></div>
    </div>
    <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:.9rem 1.4rem;display:flex;align-items:center;gap:.65rem;">
      <div style="width:7px;height:7px;border-radius:50%;background:#0ea5e9;box-shadow:0 0 6px rgba(14,165,233,.6);"></div>
      <div><div style="font-weight:600;font-size:.78rem;color:#e8f4fd;">2 Generation Modes</div><div style="font-family:'Fira Code',monospace;font-size:.58rem;color:rgba(200,216,234,.3);margin-top:1px;">Topic · Document RAG</div></div>
    </div>
    <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:.9rem 1.4rem;display:flex;align-items:center;gap:.65rem;">
      <div style="width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 6px rgba(34,197,94,.6);"></div>
      <div><div style="font-weight:600;font-size:.78rem;color:#e8f4fd;"></div><div style="font-family:'Fira Code',monospace;font-size:.58rem;color:rgba(200,216,234,.3);margin-top:1px;">HuggingFace Inference</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


st.subheader("AI-Powered Curriculum Generator")
duration = st.selectbox(
    "Program Duration (Weeks)",
    [4, 8, 12, 16, 20, 24]
)

academic_level = st.selectbox(
    "Academic Level",
    ["Foundation Level", "Undergraduate Level", "Advanced / Graduate Level"]
)

program_focus = st.selectbox(
    "Program Focus",
    ["Theoretical Emphasis", "Applied / Industry-Focused", "Research-Intensive"]
)

evaluation_framework = st.selectbox(
    "Evaluation Framework",
    ["Continuous Assessment",
     "Project-Centric Evaluation",
     "Examination + Capstone Model"]
)

mode = st.radio(
    "Select Generation Mode:",
    ["Topic-Based", "Document-Based (RAG)"],
    key="generation_mode"
)
if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode

if st.session_state.last_mode != mode:
    st.session_state.generated_output = ""
    st.session_state.last_mode = mode

# ---------------- TOPIC MODE ----------------
if mode == "Topic-Based":

    topic = st.text_input("Enter Course Title")

    if st.button("Generate Curriculum"):
        if topic:
            with st.spinner("Generating curriculum..."):
                output = generate_curriculum(topic,academic_level,duration,program_focus,evaluation_framework)
                st.session_state.generated_output = output
        else:
            st.warning("Please enter a course title.")


# ---------------- RAG MODE ----------------
elif mode == "Document-Based (RAG)":

    uploaded_file = st.file_uploader("Upload Curriculum PDF", type=["pdf"])

    if uploaded_file:
        st.success("PDF uploaded successfully!")
        import lancedb

        # ---- Extract Text ----
        def extract_text_from_uploaded(file):
            text = ""
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            return text

        # ---- Chunk Text ----
        def chunk_text(text, chunk_size=800):
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        # ---- Process Document (runs once per upload) ----
        text = extract_text_from_uploaded(uploaded_file)
        chunks = chunk_text(text)
        embeddings = model.encode(chunks)

        db = lancedb.connect("lancedb")
        table = db.create_table(
            "curriculum",
            data=[
                {"text": chunks[i], "vector": embeddings[i]}
                for i in range(len(chunks))
            ],
            mode="overwrite"
        )

        # ---- Generate Button ----
        if st.button("Generate From Document"):

            query_embedding = model.encode(
                [f"Create a structured {duration}-week academic curriculum"]
            )[0]

            results = table.search(query_embedding).limit(5).to_list()

            retrieved_texts = [r["text"] for r in results]
            context = "\n\n".join(retrieved_texts)

            prompt = f"""
            You are an academic curriculum restructuring engine.

            Use ONLY the following retrieved sections:

            {context}

            Restructure this into a structured {duration}-week academic course plan.
            Include:
            - Course Overview
            - Weekly Breakdown
            - Learning Outcomes
            - Assessment Strategy

            Do NOT add external information.
            """

            response = client.chat_completion(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )

            st.session_state.generated_output = response.choices[0].message.content
if "generated_output" in st.session_state and st.session_state.generated_output:
    st.markdown(st.session_state.generated_output)
if "generated_output" in st.session_state and st.session_state.generated_output:
    clean_text = clean_markdown(st.session_state.generated_output)
    pdf_file = generate_pdf(clean_text)

    st.download_button(
        label="📥 Download Curriculum as PDF",
        data=pdf_file,
        file_name="Curriculum.pdf",
        mime="application/pdf"
    )