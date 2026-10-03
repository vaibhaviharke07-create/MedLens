import re
import pandas as pd


# -----------------------------
# Extract batch number
# -----------------------------
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


# -----------------------------
# Load CDSCO dataset
# -----------------------------
df = pd.read_csv("data/cdsco_nsq.csv")


# -----------------------------
# Sample OCR text
# -----------------------------
ocr_text = """
LINEZOLID TABLETS IP 600mg

Batch No: TG261185

Mfg Date: May-2026
Exp Date: Apr-2028

Manufactured By:
Symbiosis Pharmaceuticals Pvt. Ltd.
"""


# -----------------------------
# Extract batch
# -----------------------------
batch_number = extract_batch_number(ocr_text)

print("Extracted batch number:", batch_number)


# -----------------------------
# Search CDSCO database
# -----------------------------
if batch_number:

    result = df[
        df["batch_no"]
        .astype(str)
        .str.strip()
        .str.upper()
        == batch_number
    ]

    if not result.empty:

        print("\n⚠️ OFFICIAL CDSCO ALERT MATCH FOUND")

        print(
            result.to_string(index=False)
        )

        print(
            "\nPlease verify this medicine with "
            "a pharmacist or appropriate healthcare professional."
        )

    else:

        print(
            "\n✅ No matching batch found "
            "in the MedLens CDSCO dataset."
        )

else:

    print(
        "\n❓ Could not confidently identify "
        "the batch number."
    )