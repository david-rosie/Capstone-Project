import streamlit as st
from io import BytesIO
import pdfplumber

st.set_page_config(page_title="PDF Extractor", layout="wide")
st.title("PDF Text Extractor")

st.write("Upload a PDF file and extract its text content.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.read()

        with pdfplumber.open(BytesIO(bytes_data)) as pdf:
            st.sidebar.header("Document Info")
            st.sidebar.write("Pages:", len(pdf.pages))

            metadata = pdf.metadata
            if metadata:
                st.sidebar.write("Title:", metadata.get("Title", ""))
                st.sidebar.write("Author:", metadata.get("Author", ""))
                st.sidebar.write("Subject:", metadata.get("Subject", ""))
                st.sidebar.write("Producer:", metadata.get("Producer", ""))

            num_pages = len(pdf.pages)
            selected_pages = st.slider("Select pages to extract", 1, num_pages, (1, min(5, num_pages)))
            if st.button("Extract text"):
                start_page, end_page = selected_pages
                extracted_text = []
                for p in range(start_page - 1, end_page):
                    page = pdf.pages[p]
                    extracted_text.append(page.extract_text() or "")

                full_text = "\n\n".join(extracted_text)

                st.subheader("Extracted text")
                st.text_area("PDF text", value=full_text, height=400)

                st.download_button(
                    label="Download extracted text",
                    data=full_text,
                    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_extracted.txt",
                    mime="text/plain",
                )

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
