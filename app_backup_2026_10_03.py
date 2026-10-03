
import os
import re
import time

import pandas as pd
import requests
import streamlit as st
import pytesseract

from PIL import Image
from dotenv import load_dotenv
from rapidfuzz.fuzz import token_set_ratio


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(text):
    """Normalize text for comparison."""

    if text is None:
        return ""

    text = str(text).upper().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def extract_batch_number(text):
    """Extract batch or lot number from OCR text."""

    patterns = [
        r"Batch\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"B\.?\s*No\.?\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"Lot\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Z0-9\-]+)",
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


def find_best_match(text, values, threshold=55):
    """Find the best fuzzy match."""

    text = normalize_text(text)

    if not text:
        return None, 0

    best_value = None
    best_score = 0

    for value in values:

        value = str(value)

        score = token_set_ratio(
            text,
            normalize_text(value)
        )

        if score > best_score:
            best_score = score
            best_value = value

    if best_score >= threshold:
        return best_value, best_score

    return None, best_score


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedLens",
    page_icon="MedLens",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .medlens-header {
        text-align: center;
        padding: 25px 10px 20px 10px;
    }

    .medlens-logo {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .medlens-tagline {
        font-size: 21px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .medlens-description {
        max-width: 750px;
        margin: auto;
        font-size: 16px;
        line-height: 1.6;
    }

    .report-title {
        font-size: 28px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 5px;
    }

    .report-subtitle {
        font-size: 15px;
        margin-bottom: 20px;
    }

    .ai-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 10px;
        border: 1px solid #cccccc;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="medlens-header">

        <div class="medlens-logo">
            MedLens
        </div>

        <div class="medlens-tagline">
            Scan. Understand. Stay Informed.
        </div>

        <div class="medlens-description">
            An AI-powered medicine information and verification assistant
            that helps users understand medicine labels and check identified
            batch information against the CDSCO dataset included in MedLens.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INFORMATION
# ============================================================

st.info(
    "Upload a clear medicine label or prescription image. "
    "MedLens will extract readable information, identify the batch number "
    "when possible, check the included CDSCO dataset, and provide a simple AI explanation."
)


# ============================================================
# LANGUAGE
# ============================================================

language = st.selectbox(
    "Explanation Language",
    [
        "English",
        "Hindi",
        "Marathi"
    ]
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("### Scan Medicine Information")

st.write(
    "Upload a clear medicine label or prescription image."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    st.success("Image uploaded successfully.")

    try:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Medicine Image",
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Unable to open the uploaded image: {e}"
        )

        st.stop()


    # ========================================================
    # TESSERACT
    # ========================================================

    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if os.path.exists(tesseract_path):

        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    else:

        st.error(
            "Tesseract OCR was not found at: "
            + tesseract_path
        )

        st.stop()


    # ========================================================
    # OCR
    # ========================================================

    st.markdown("### Extracting Information")

    with st.spinner("Reading medicine information..."):

        try:

            extracted_text = pytesseract.image_to_string(
                image
            )

        except Exception as e:

            st.error(
                f"OCR failed: {e}"
            )

            st.stop()


    extracted_text = extracted_text.strip()


    # ========================================================
    # OCR RESULT
    # ========================================================

    if not extracted_text:

        st.warning(
            "No readable text was detected. "
            "Please upload a clearer medicine label image."
        )

        st.stop()


    st.success(
        "Medicine text extracted successfully."
    )


    with st.expander("View Extracted Text"):

        st.text(
            extracted_text
        )


    # ========================================================
    # INITIAL VARIABLES
    # ========================================================

    batch_number = extract_batch_number(
        extracted_text
    )

    detected_product = None
    detected_manufacturer = None

    product_score = 0
    manufacturer_score = 0

    batch_result = pd.DataFrame()


    # ========================================================
    # CDSCO DATASET
    # ========================================================

    cdsco_file = "data/cdsco_nsq.csv"


    if not os.path.exists(cdsco_file):

        st.error(
            "CDSCO dataset was not found. "
            "Make sure data/cdsco_nsq.csv exists."
        )

    else:

        try:

            df = pd.read_csv(
                cdsco_file
            )

            st.success(
                f"CDSCO dataset loaded successfully. "
                f"Records: {len(df)}"
            )

        except Exception as e:

            st.error(
                f"Unable to load CDSCO dataset: {e}"
            )

            df = pd.DataFrame()


        # ====================================================
        # CDSCO VERIFICATION
        # ====================================================

        if not df.empty:


            # ------------------------------------------------
            # PRODUCT MATCHING
            # ------------------------------------------------

            if "product_name" in df.columns:

                detected_product, product_score = find_best_match(
                    extracted_text,
                    df["product_name"].dropna().unique(),
                    threshold=55
                )


            # ------------------------------------------------
            # MANUFACTURER MATCHING
            # ------------------------------------------------

            if "manufacturer" in df.columns:

                detected_manufacturer, manufacturer_score = find_best_match(
                    extracted_text,
                    df["manufacturer"].dropna().unique(),
                    threshold=55
                )


            # ------------------------------------------------
            # BATCH VERIFICATION
            # ------------------------------------------------

            if batch_number:

                if "batch_no" not in df.columns:

                    st.warning(
                        "The CDSCO dataset does not contain "
                        "a batch_no column."
                    )

                else:

                    batch_result = df[
                        df["batch_no"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        ==
                        batch_number.strip().upper()
                    ]


                    # ========================================
                    # MATCH FOUND
                    # ========================================

                    if not batch_result.empty:

                        st.error(
                            "Matching batch found in the "
                            "included CDSCO NSQ dataset."
                        )

                        st.write(
                            "The identified batch matches an "
                            "official quality-alert record included "
                            "in the MedLens dataset."
                        )

                        st.dataframe(
                            batch_result,
                            use_container_width=True
                        )

                        st.warning(
                            "This result does not by itself determine "
                            "whether the medicine in your possession is "
                            "safe or authentic. Verify the product, batch "
                            "and manufacturer details with a pharmacist, "
                            "manufacturer or appropriate authority."
                        )


                    # ========================================
                    # NO MATCH
                    # ========================================

                    else:

                        st.success(
                            "No matching batch number was found "
                            "in the included CDSCO dataset."
                        )

                        st.write(
                            "No matching record was found for this "
                            "batch in the local CDSCO dataset used "
                            "by this demo."
                        )

                        if detected_product:

                            st.write(
                                f"Possible Product Match: "
                                f"{detected_product}"
                            )

                            st.write(
                                f"Product Similarity: "
                                f"{product_score}%"
                            )


                        if detected_manufacturer:

                            st.write(
                                f"Possible Manufacturer Match: "
                                f"{detected_manufacturer}"
                            )

                            st.write(
                                f"Manufacturer Similarity: "
                                f"{manufacturer_score}%"
                            )


                        st.info(
                            "No-match only means that this batch was "
                            "not found in the included dataset. "
                            "It does NOT prove that the medicine is "
                            "safe or authentic."
                        )


            # ------------------------------------------------
            # NO BATCH NUMBER
            # ------------------------------------------------

            else:

                st.warning(
                    "A batch number could not be confidently "
                    "extracted from the image."
                )

                st.write(
                    "Please manually check the batch number on "
                    "the medicine packaging and verify it with "
                    "a pharmacist or appropriate authority."
                )


    # ========================================================
    # VERIFICATION REPORT
    # ========================================================

    st.markdown("---")

    st.markdown(
        """
        <div class="report-title">
            MedLens Verification Report
        </div>

        <div class="report-subtitle">
            Automated information extraction and dataset-based verification
        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Batch Number",
            batch_number if batch_number else "Not detected"
        )


    with col2:

        if not batch_result.empty:

            verification_status = "CDSCO Match"

        elif batch_number:

            verification_status = "No Dataset Match"

        else:

            verification_status = "Manual Check"

        st.metric(
            "Verification Status",
            verification_status
        )


    st.write("### Verification Details")


    if batch_number:

        st.write(
            f"Detected Batch Number: `{batch_number}`"
        )

    else:

        st.write(
            "Detected Batch Number: Not confidently detected"
        )


    if detected_product:

        st.write(
            f"Possible Product: {detected_product}"
        )

        st.write(
            f"Product Match Score: {product_score}%"
        )


    if detected_manufacturer:

        st.write(
            f"Possible Manufacturer: {detected_manufacturer}"
        )

        st.write(
            f"Manufacturer Match Score: {manufacturer_score}%"
        )


    # ========================================================
    # HUMAN VERIFICATION
    # ========================================================

    st.write(
        "### Human Verification Checklist"
    )


    st.checkbox(
        "I checked the medicine name on the physical package."
    )

    st.checkbox(
        "I checked the batch number on the physical package."
    )

    st.checkbox(
        "I checked the manufacturer information."
    )

    st.checkbox(
        "I will verify important information with a pharmacist or doctor."
    )


    # ========================================================
    # GEMINI AI
    # ========================================================

    st.markdown("---")

    st.markdown(
        """
        <div class="ai-badge">
            AI MEDICINE EXPLANATION
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "### Simple AI Explanation"
    )


    # ========================================================
    # GEMINI API KEY
    # ========================================================

    api_key = None


    try:

        api_key = st.secrets.get(
            "GEMINI_API_KEY"
        )

    except Exception:

        api_key = None


    if not api_key:

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )


    if api_key:

        api_key = str(api_key)

        api_key = api_key.replace(
            "\ufeff",
            ""
        )

        api_key = api_key.strip()


    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    if not api_key:

        st.warning(
            "Gemini API key is not configured. "
            "CDSCO verification is still available."
        )

    else:

        prompt = f"""
You are MedLens, an informational medicine-label assistant.

The user uploaded a medicine label or prescription image.

OCR text extracted from the image:

--------------------
{extracted_text}
--------------------

The selected explanation language is:

{language}

Provide a simple and clear informational explanation using these sections:

1. Medicine Name
2. Strength / Dosage Information
3. Instructions written on the label
4. Frequency / Timing if clearly present
5. Warnings / Precautions if clearly present
6. Simple Explanation
7. Information Not Clear

IMPORTANT SAFETY RULES:

- Do not diagnose any disease.
- Do not prescribe medicine.
- Do not recommend starting medicine.
- Do not recommend stopping medicine.
- Do not change dosage.
- Do not calculate dosage.
- Do not invent missing information.
- Do not assume information that is unclear.
- Clearly state when information cannot be read.
- If dosage or instructions are unclear, tell the user to verify them with a doctor or pharmacist.
- This is an informational explanation only.
"""


        url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models/gemini-3.8-flash:generateContent"
        )


        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }


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


        response = None


        # ====================================================
        # RETRY GEMINI REQUEST
        # ====================================================

        for attempt in range(3):

            try:

                with st.spinner(
                    "AI is preparing a simple explanation..."
                ):

                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=180
                    )


                if response.status_code == 503:

                    if attempt < 2:

                        time.sleep(
                            5 * (attempt + 1)
                        )

                        continue

                break


            except requests.exceptions.Timeout:

                if attempt == 2:

                    st.error(
                        "Gemini request timed out. "
                        "Please try again."
                    )


            except requests.exceptions.RequestException as e:

                if attempt == 2:

                    st.error(
                        f"Gemini connection error: {e}"
                    )


        # ====================================================
        # GEMINI RESPONSE
        # ====================================================

        if response is not None:


            if response.status_code == 200:

                try:

                    result = response.json()

                    candidates = result.get(
                        "candidates",
                        []
                    )


                    if candidates:

                        content = candidates[0].get(
                            "content",
                            {}
                        )


                        parts = content.get(
                            "parts",
                            []
                        )


                        if parts:

                            ai_text = parts[0].get(
                                "text",
                                ""
                            )


                            if ai_text:

                                st.markdown(
                                    ai_text
                                )

                            else:

                                st.warning(
                                    "Gemini returned an empty explanation."
                                )

                        else:

                            st.warning(
                                "Gemini did not return readable content."
                            )

                    else:

                        st.warning(
                            "Gemini did not return a response."
                        )


                except Exception as e:

                    st.error(
                        f"Unable to process Gemini response: {e}"
                    )


            elif response.status_code == 400:

                st.error(
                    "Gemini API request was rejected. "
                    "Please check that your API key is valid."
                )

                with st.expander("Technical Details"):

                    st.code(
                        response.text
                    )


            elif response.status_code == 401:

                st.error(
                    "Gemini API key is invalid or unauthorized."
                )


            elif response.status_code == 403:

                st.error(
                    "Gemini API key does not have permission "
                    "to use this API."
                )


            elif response.status_code == 404:

                st.error(
                    "The Gemini model endpoint was not found."
                )

                with st.expander("Technical Details"):

                    st.code(
                        response.text
                    )


            elif response.status_code == 503:

                st.error(
                    "Gemini is temporarily overloaded. "
                    "Please try again later."
                )


            else:

                st.error(
                    f"Gemini API error: HTTP {response.status_code}"
                )

                with st.expander("Technical Details"):

                    st.code(
                        response.text
                    )


# ============================================================
# MEDICAL SAFETY NOTICE
# ============================================================

st.markdown("---")

st.warning(
    "Medical Safety Notice: MedLens is an informational assistant "
    "and is not a substitute for a doctor or pharmacist. "
    "Do not change, stop, or start any medicine based only on this app. "
    "Always verify important medical information with a qualified "
    "healthcare professional."
)
