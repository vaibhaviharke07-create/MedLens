import streamlit as st

st.set_page_config(
    page_title="MedLens",
    layout="centered"
)

# Load CSS
with open("styles.css", "r", encoding="utf-8") as file:
    css = file.read()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)

# Load HTML
with open("header.html", "r", encoding="utf-8") as file:
    header = file.read()

st.markdown(
    header,
    unsafe_allow_html=True
)

st.write("UI test successful.")