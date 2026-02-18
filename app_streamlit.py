import streamlit as st
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

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
    learning outcome for the course and relatable job roles: (upto 2 lines ) 
    Duration for the course: (in weeks)

    Course Name:
    Level: (Intermediate)
    topic : (upto 3 topics)
    learning outcome for the course and relatable job roles: (upto 2 lines ) 
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

st.title("🎓 CurricuForge")
st.subheader("AI-Powered Curriculum Generator")

topic = st.text_input("Enter Course Title")

if st.button("Generate Curriculum"):
    if topic:
        with st.spinner("Generating curriculum..."):
            output = generate_curriculum(topic)
            st.markdown(output)
    else:
        st.warning("Please enter a course title.")
