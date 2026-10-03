import streamlit as st
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import requests
import os
import time
import re
import pandas as pd
from rapidfuzz.fuzz import token_set_ratio


# =====================================================
# TEXT NORMALIZATION
# =====================================================

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =====================================================
# BATCH NUMBER EXTRACTION
# =====================================================

def extract_batch_number(text):

    patterns = [
        r"Batch\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"B\.?\s*No\.?\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"Lot\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Z0-9\-]+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip().upper()

    return None


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MedLens",
    page_icon="💊",
    layout="centered"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    .stButton > button {
        width: 100%;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# LOAD ENVIRONMENT
# =====================================================

load_dotenv()


# =====================================================
# GEMINI API KEY
# =====================================================

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:

    st.error(
        "Gemini API key is not configured in Streamlit Secrets."
    )

    st.stop()


api_key = str(api_key).replace("\ufeff", "").strip()


# =====================================================
# TESSERACT OCR
# =====================================================

tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if not os.path.exists(tesseract_path):

    st.error(
        "Tesseract OCR was not found at: "
        + tesseract_path
    )

    st.stop()


pytesseract.pytesseract.tesseract_cmd = tesseract_path


# =====================================================
# TITLE
# =====================================================

st.title("💊 MedLens")

st.caption(
    "Scan. Understand. Stay Informed."
)

st.subheader(
    "AI-Powered Medicine Information Assistant"
)

st.write(
    "Upload a medicine label or prescription image "
    "to extract and understand the information."
)

st.info(
    "💡 Upload a clear image of a medicine label or "
    "prescription to extract and understand the information."
)


# =====================================================
# LANGUAGE
# =====================================================

language = st.selectbox(
    "🌐 Choose explanation language",
    ["English", "Hindi", "Marathi"]
)


# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "📷 Upload a medicine label or prescription",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # =================================================
    # OPEN IMAGE
    # =================================================

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Medicine Image",
        width="stretch"
    )

    st.success(
        "Image uploaded successfully! ✅"
    )


    # =================================================
    # OCR
    # =================================================

    with st.spinner("🔍 Reading the text..."):

        extracted_text = pytesseract.image_to_string(
            image
        )


    st.subheader("📄 Extracted Text")


    if extracted_text.strip():

        st.text_area(
            "Text detected from the image:",
            extracted_text,
            height=200
        )


        # =================================================
        # INITIAL VARIABLES
        # =================================================

        batch_number = extract_batch_number(
            extracted_text
        )

        detected_product = None
        detected_manufacturer = None

        product_score = 0
        manufacturer_score = 0

        batch_result = pd.DataFrame()


        # =================================================
        # CDSCO VERIFICATION
        # =================================================

        st.subheader(
            "🏛️ CDSCO Safety Alert Check"
        )


        if batch_number:

            st.write(
                f"**Batch number detected:** `{batch_number}`"
            )


            try:

                # =========================================
                # LOAD DATASET
                # =========================================

                cdsco_df = pd.read_csv(
                    "data/cdsco_nsq.csv"
                )


                # =========================================
                # PRODUCT DETECTION
                # =========================================

                for product in cdsco_df[
                    "product_name"
                ].dropna().unique():

                    score = token_set_ratio(
                        normalize_text(product),
                        normalize_text(extracted_text)
                    )

                    if score > product_score:

                        product_score = score

                        detected_product = str(
                            product
                        )


                if product_score < 70:

                    detected_product = None


                # =========================================
                # MANUFACTURER DETECTION
                # =========================================

                for manufacturer in cdsco_df[
                    "manufacturer"
                ].dropna().unique():

                    score = token_set_ratio(
                        normalize_text(manufacturer),
                        normalize_text(extracted_text)
                    )

                    if score > manufacturer_score:

                        manufacturer_score = score

                        detected_manufacturer = str(
                            manufacturer
                        )


                if manufacturer_score < 70:

                    detected_manufacturer = None


                # =========================================
                # SEARCH BATCH
                # =========================================

                batch_result = cdsco_df[
                    cdsco_df["batch_no"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    ==
                    batch_number.strip().upper()
                ]


                # =========================================
                # BATCH FOUND
                # =========================================

                if not batch_result.empty:

                    st.error(
                        "⚠️ This batch matches an official "
                        "CDSCO NSQ record in the dataset "
                        "used by MedLens."
                    )


                    st.write(
                        "### 🔎 Official CDSCO Record"
                    )


                    st.dataframe(
                        batch_result,
                        use_container_width=True
                    )


                    # =====================================
                    # PRODUCT COMPARISON
                    # =====================================

                    official_product = str(
                        batch_result.iloc[0][
                            "product_name"
                        ]
                    )


                    if detected_product:

                        product_match_score = token_set_ratio(
                            normalize_text(
                                detected_product
                            ),
                            normalize_text(
                                official_product
                            )
                        )


                        if product_match_score >= 70:

                            st.success(
                                f"✅ Product name matches "
                                f"the CDSCO record "
                                f"({product_match_score}% similarity)."
                            )

                        else:

                            st.warning(
                                "⚠️ Product name could not "
                                "be confidently matched "
                                "with the CDSCO record."
                            )

                    else:

                        st.info(
                            "ℹ️ Product name could not be "
                            "confidently identified from "
                            "the OCR text."
                        )


                    # =====================================
                    # MANUFACTURER COMPARISON
                    # =====================================

                    official_manufacturer = str(
                        batch_result.iloc[0][
                            "manufacturer"
                        ]
                    )


                    if detected_manufacturer:

                        manufacturer_match_score = token_set_ratio(
                            normalize_text(
                                detected_manufacturer
                            ),
                            normalize_text(
                                official_manufacturer
                            )
                        )


                        if manufacturer_match_score >= 70:

                            st.success(
                                f"✅ Manufacturer matches "
                                f"the CDSCO record "
                                f"({manufacturer_match_score}% similarity)."
                            )

                        else:

                            st.warning(
                                "⚠️ Manufacturer could not "
                                "be confidently matched "
                                "with the CDSCO record."
                            )

                    else:

                        st.info(
                            "ℹ️ Manufacturer could not be "
                            "confidently identified from "
                            "the OCR text."
                        )


                    st.warning(
                        "Please verify this information "
                        "with a pharmacist, manufacturer, "
                        "or appropriate healthcare professional."
                    )


                # =========================================
                # BATCH NOT FOUND
                # =========================================

                else:

                    st.success(
                        "✅ No matching batch was found "
                        "in the CDSCO dataset used by MedLens."
                    )


                    if detected_product:

                        st.write(
                            f"**Product identified:** "
                            f"{detected_product}"
                        )

                        st.caption(
                            f"Product similarity score: "
                            f"{product_score}%"
                        )

                    else:

                        st.info(
                            "Product name could not be "
                            "confidently identified."
                        )


                    if detected_manufacturer:

                        st.write(
                            f"**Manufacturer identified:** "
                            f"{detected_manufacturer}"
                        )

                        st.caption(
                            f"Manufacturer similarity score: "
                            f"{manufacturer_score}%"
                        )

                    else:

                        st.info(
                            "Manufacturer could not be "
                            "confidently identified."
                        )


                    st.info(
                        "A no-match result does NOT prove "
                        "that the medicine is safe, genuine, "
                        "or authentic. It only means that "
                        "no matching batch was found in the "
                        "CDSCO dataset currently included "
                        "in MedLens."
                    )


            except Exception as e:

                st.warning(
                    "CDSCO verification could not be completed."
                )

                st.code(
                    str(e)
                )


        else:

            st.warning(
                "❓ MedLens could not confidently identify "
                "a batch number from the uploaded image."
            )

            st.info(
                "Please verify the batch number manually "
                "from the medicine package or with a pharmacist."
            )


        # =================================================
        # MEDLENS VERIFICATION REPORT
        # =================================================

        st.subheader(
            "🛡️ MedLens Verification Report"
        )

        st.caption(
            "A structured summary of the information "
            "detected by MedLens."
        )


        # ================================================
        # REPORT VALUES
        # ================================================

        report_product = (
            detected_product
            if detected_product
            else "Not clearly identified"
        )


        report_manufacturer = (
            detected_manufacturer
            if detected_manufacturer
            else "Not clearly identified"
        )


        report_batch = (
            batch_number
            if batch_number
            else "Not clearly identified"
        )


        # ================================================
        # REPORT CARDS
        # ================================================

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "💊 Medicine",
                report_product
            )

            st.metric(
                "🏷️ Batch Number",
                report_batch
            )


        with col2:

            st.metric(
                "🏭 Manufacturer",
                report_manufacturer
            )


            if batch_number and not batch_result.empty:

                st.metric(
                    "🔎 CDSCO Record",
                    "Match Found"
                )

            elif batch_number:

                st.metric(
                    "🔎 CDSCO Record",
                    "No Match"
                )

            else:

                st.metric(
                    "🔎 CDSCO Record",
                    "Could Not Check"
                )


        # ================================================
        # VERIFICATION STATUS
        # ================================================

        st.write("### 🔎 Verification Status")


        if batch_number and not batch_result.empty:

            st.error(
                "⚠️ Official CDSCO NSQ record match found."
            )


            official_result = str(
                batch_result.iloc[0][
                    "nsq_result"
                ]
            )


            st.write(
                f"**Official NSQ Result:** {official_result}"
            )


            st.info(
                "This means the detected batch matches "
                "an official NSQ record included in the "
                "MedLens dataset. Verify the physical "
                "medicine package and important details "
                "with a pharmacist, manufacturer, or "
                "appropriate authority."
            )


        elif batch_number:

            st.success(
                "ℹ️ No matching batch was found in the "
                "CDSCO dataset currently included in MedLens."
            )


            st.info(
                "A no-match result does NOT prove that the "
                "medicine is safe, genuine, or authentic. "
                "It only means that no matching batch was "
                "found in the dataset used by MedLens."
            )


        else:

            st.warning(
                "⚠️ Batch number could not be confidently identified."
            )


            st.info(
                "Manual verification of the batch number is required."
            )


        # ================================================
        # HUMAN VERIFICATION CHECKLIST
        # ================================================

        st.write(
            "### 👨‍⚕️ Human Verification Checklist"
        )


        check1 = (
            "✅"
            if report_product != "Not clearly identified"
            else "⚠️"
        )


        check2 = (
            "✅"
            if report_batch != "Not clearly identified"
            else "⚠️"
        )


        check3 = (
            "✅"
            if report_manufacturer != "Not clearly identified"
            else "⚠️"
        )


        st.write(
            f"{check1} Medicine name identified"
        )

        st.write(
            f"{check2} Batch number identified"
        )

        st.write(
            f"{check3} Manufacturer identified"
        )


        st.warning(
            "⚠️ Important medical information should be "
            "confirmed with a qualified healthcare professional."
        )


        # =================================================
        # GEMINI AI ANALYSIS
        # =================================================

        st.subheader(
            "🤖 MedLens AI Analysis"
        )

        st.caption(
            "AI-generated explanation based only "
            "on the uploaded text."
        )


        # =================================================
        # GEMINI PROMPT
        # =================================================

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


        # =================================================
        # GEMINI API
        # =================================================

        with st.spinner(
            "🤖 Understanding the medicine information..."
        ):

            url = (
                "https://generativelanguage.googleapis.com/"
                "v1beta/models/gemini-3.8-flash:generateContent"
            )


            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }


            try:

                response = None


                for attempt in range(3):

                    response = requests.post(
                        url,
                        headers={
                            "x-goog-api-key": api_key,
                            "Content-Type": "application/json"
                        },
                        json=payload,
                        timeout=180
                    )


                    if response.status_code == 200:

                        break


                    if response.status_code == 503:

                        if attempt < 2:

                            wait_time = 5 * (
                                attempt + 1
                            )

                            st.warning(
                                f"Gemini is temporarily busy. "
                                f"Retrying in {wait_time} seconds..."
                            )

                            time.sleep(
                                wait_time
                            )

                        else:

                            st.error(
                                "Gemini is currently experiencing "
                                "high demand. Please try again "
                                "in a few moments."
                            )

                            st.stop()


                    else:

                        st.error(
                            f"Gemini API error: "
                            f"{response.status_code}"
                        )

                        st.code(
                            response.text
                        )

                        st.stop()


                # =========================================
                # READ RESPONSE
                # =========================================

                result = response.json()

                answer = (
                    result["candidates"][0]
                    ["content"]["parts"][0]["text"]
                )


                st.markdown(
                    answer
                )


            except requests.exceptions.Timeout:

                st.error(
                    "Gemini took too long to respond. "
                    "Please try again."
                )


            except requests.exceptions.RequestException as e:

                st.error(
                    "Could not connect to the Gemini API."
                )

                st.code(
                    str(e)
                )


            except (KeyError, IndexError):

                st.error(
                    "Gemini returned an unexpected response."
                )

                if response is not None:

                    st.code(
                        response.text
                    )


    else:

        st.warning(
            "No readable text was detected. "
            "Please upload a clearer medicine image."
        )


# =====================================================
# MEDICAL SAFETY NOTICE
# =====================================================

st.warning(
    "⚠️ Medical Safety Notice: MedLens is an informational "
    "assistant and is not a substitute for a doctor or pharmacist. "
    "Do not change, stop, or start any medicine based only on this app. "
    "Always verify important medical information with a qualified "
    "healthcare professional."
)