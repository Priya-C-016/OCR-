import streamlit as st
from PIL import Image
from ocr_model import extract_text

st.title("OCR App for Hindi and English Text Extraction")
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    extracted_text = extract_text(uploaded_file)
    st.write("Extracted Text:")
    st.text(extracted_text)
    keyword = st.text_input("Enter a keyword to search in the extracted text:")
    if keyword:
        matches = [word for word in extracted_text.split() if keyword.lower() in word.lower()]
        st.write(f"Found {len(matches)} matches for '{keyword}':")
        st.write(", ".join(matches))
