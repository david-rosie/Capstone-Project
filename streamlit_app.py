import streamlit as st
from io import BytesIO
import pdfplumber
import pickle
import pandas as pd

st.set_page_config(page_title="Multi-Tool App", layout="wide")
st.title("Multi-Tool App")

tab1, tab2 = st.tabs(["PDF Extractor", "Breast Cancer Predictor"])

with tab1:
    st.header("PDF Text Extractor")
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

with tab2:
    st.header("Breast Cancer Diagnosis Predictor")
    st.write("Upload a CSV file with one row of 30 numerical features for prediction.")
    st.write("Features order: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst")

    uploaded_csv = st.file_uploader("Upload CSV for prediction", type=["csv"], key="csv")

    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            if df.shape[0] != 1 or df.shape[1] != 30:
                st.error("CSV must have exactly 1 row and 30 columns.")
            else:
                # Load model and scaler
                with open('breast_cancer_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)

                # Scale the input
                scaled_input = scaler.transform(df.values)

                # Predict
                prediction = model.predict(scaled_input)
                prob = model.predict_proba(scaled_input)

                st.subheader("Prediction Result")
                st.write("Diagnosis:", "Malignant" if prediction[0] == 1 else "Benign")
                st.write("Note: This is a machine learning prediction and should not be relied upon as a medical diagnosis. Always consult a healthcare professional for medical advice.")

        except Exception as e:
            st.error(f"Error processing prediction: {e}")
