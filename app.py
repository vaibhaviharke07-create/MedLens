import streamlit as st
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import os
from google import genai
st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }

    .stButton>button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key is not configured.")
    st.stop()

api_key = api_key.strip()

client = genai.Client(api_key=api_key)
# Tesseract location
import shutil

tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Page configuration
st.set_page_config(
    page_title="MedLens",
    page_icon="💊",
    layout="centered"
)

# Title
st.title("💊 MedLens")
st.caption("Scan. Understand. Stay Informed.")
st.subheader("AI-Powered Medicine Information Assistant")


st.write(
    "Upload a medicine label or prescription image "
    "to extract and understand the information."
)
st.info(
    "💡 Upload a clear image of a medicine label or prescription "
    "to extract and understand the information."
)
language = st.selectbox(
    "🌐 Choose explanation language",
    ["English", "Hindi", "Marathi"]
)

# Upload image
uploaded_file = st.file_uploader(
    "📷 Upload a medicine label or prescription",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Medicine Image",
        use_container_width=True
    )

    st.success("Image uploaded successfully! ✅")

    # OCR
    with st.spinner("🔍 Reading the text..."):
        extracted_text = pytesseract.image_to_string(image)

    st.subheader("📄 Extracted Text")

    if extracted_text.strip():

        st.text_area(
            "Text detected from the image:",
            extracted_text,
            height=200
        )

        # Gemini analysis
        st.subheader("🤖 MedLens AI Analysis")
        st.caption("AI-generated explanation based only on the uploaded text.")

        with st.spinner("🤖 Understanding the medicine information..."):

            prompt = f"""
You are MedLens, an AI-powered medicine information assistant.
Respond in {language}.

Your job is to explain ONLY the information that can be identified
from the uploaded medicine label or prescription.

OCR TEXT:
{extracted_text}

Return the answer in this exact structure:

### 💊 Medicine Name
Give the medicine name if clearly available.

### ⚖️ Strength / Dosage
Give the strength or dosage exactly as written.
Do not calculate or change it.

### 🕐 Instructions
Explain the instructions written on the label or prescription.

### ⏰ Frequency / Timing
Mention frequency or timing only if clearly present.

### ⚠️ Warnings / Precautions
Mention warnings or precautions that are explicitly available
in the extracted text.

### 💡 Simple Explanation
Explain the available information in easy language.

### ❓ Information Not Clear
Mention anything that could not be read or identified clearly.

IMPORTANT SAFETY RULES:
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not recommend changing, stopping, or starting medication.
- Do not invent missing information.
- Do not guess unclear text.
- Never calculate a dosage.
- If the OCR text is unclear, clearly say so.
- Tell the user to confirm important or unclear information
  with a doctor or pharmacist.
"""

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

        st.markdown(response.text)

    else:

        st.warning(
            "No readable text was detected. "
            "Please upload a clearer medicine image."
        )

# Safety notice
st.warning(
    "⚠️ Medical Safety Notice: MedLens is an informational assistant "
    "and is not a substitute for a doctor or pharmacist. "
    "Do not change, stop, or start any medicine based only on this app. "
    "Always verify important medical information with a qualified healthcare professional."
)