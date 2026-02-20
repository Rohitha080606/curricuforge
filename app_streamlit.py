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


def generate_curriculum(topic):

    prompt = f"""
    Create a professional university-level curriculum for a course titled '{topic}'.

    Format clearly with:

    Course Name:
    Level: (Beginner)
    topic : (upto 3 topics)
    learning outcome for the course 
    relatable job roles: (upto 2 lines ) 
    Duration for the course: (in weeks)

    Course Name:
    Level: (Intermediate)
    topic : (upto 3 topics)
    learning outcome for the course : (upto 2 lines)
    relatable job roles: (upto 2 lines ) 
    Duration for the course: (in weeks)

    Course Name:
    Level: (Advanced)
    topic : (upto 3 topics)
    learning outcome for the course: (upto 2 lines )
    relatable job roles: (upto 2 roles)
    Duration for the course: (in weeks)

    Constraints:
    No short description
    Stick to the format
    No repeatitions


    """

    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )

    return response.choices[0].message.content


# Streamlit UI
st.set_page_config(page_title="CurricuForge MVP")
st.set_page_config(
    page_title="CurricuForge",
    layout="wide"
)


st.title(" 🎓CurricuForge")
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
                output = generate_curriculum(topic)
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