import re


def extract_batch_number(text):
    patterns = [
        r"Batch\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"B\.?\s*No\.?\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"Lot\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Z0-9\-]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).strip().upper()

    return None


# Sample OCR text
ocr_text = """
LINEZOLID TABLETS IP 600mg
Batch No: TG261185
Mfg Date: May-2026
Exp Date: Apr-2028
Manufactured By: Symbiosis Pharmaceuticals Pvt. Ltd.
"""

batch = extract_batch_number(ocr_text)

if batch:
    print("✅ Batch number found:", batch)
else:
    print("❌ Batch number not found.")