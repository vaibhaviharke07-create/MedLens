from pathlib import Path

path = Path("app.py")
text = path.read_text(encoding="utf-8")

start = text.index("# ============================================================\n# CUSTOM CSS")
header_start = text.index("# ============================================================\n# HEADER", start)

header_end_marker = "    unsafe_allow_html=True\n)"
header_end = text.index(header_end_marker, header_start) + len(header_end_marker)

replacement = '''# ============================================================
# LOAD CSS
# ============================================================

with open("styles.css", "r", encoding="utf-8") as file:
    css = file.read()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# ============================================================
# LOAD HEADER
# ============================================================

with open("header.html", "r", encoding="utf-8") as file:
    header = file.read()

st.markdown(
    header,
    unsafe_allow_html=True
)'''

text = text[:start] + replacement + text[header_end:]

path.write_text(text, encoding="utf-8")

print("CSS and header separation completed successfully.")
