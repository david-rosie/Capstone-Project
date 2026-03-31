import streamlit as st
from io import BytesIO
import pdfplumber
import pickle
import pandas as pd
import os
import json
import re
from dotenv import load_dotenv
from mistralai.client import Mistral

# Load environment variables from .env file
load_dotenv()

# Feature descriptions dictionary
FEATURE_DESCRIPTIONS = {
    'radius_mean': 'Mean of distances from center to points on the perimeter',
    'texture_mean': 'Standard deviation of gray-scale values',
    'perimeter_mean': 'Mean size of the core tumor',
    'area_mean': 'No description available',
    'smoothness_mean': 'Mean of local variation in radius lengths',
    'compactness_mean': 'Mean of perimeter^2 / area - 1.0',
    'concavity_mean': 'Mean of severity of concave portions of the contour',
    'concave_points_mean': 'Mean for number of concave portions of the contour',
    'symmetry_mean': 'No description available',
    'fractal_dimension_mean': 'Mean for "coastline approximation" - 1',
    'radius_se': 'Standard error for the mean of distances from center to points on the perimeter',
    'texture_se': 'Standard error for standard deviation of gray-scale values',
    'perimeter_se': 'No description available',
    'area_se': 'No description available',
    'smoothness_se': 'Standard error for local variation in radius lengths',
    'compactness_se': 'Standard error for perimeter^2 / area - 1.0',
    'concavity_se': 'Standard error for severity of concave portions of the contour',
    'concave_points_se': 'Standard error for number of concave portions of the contour',
    'symmetry_se': 'No description available',
    'fractal_dimension_se': 'Standard error for "coastline approximation" - 1',
    'radius_worst': '"Worst" or largest mean value for mean of distances from center to points on the perimeter',
    'texture_worst': '"Worst" or largest mean value for standard deviation of gray-scale values',
    'perimeter_worst': 'No description available',
    'area_worst': 'No description available',
    'smoothness_worst': '"Worst" or largest mean value for local variation in radius lengths',
    'compactness_worst': '"Worst" or largest mean value for perimeter^2 / area - 1.0',
    'concavity_worst': '"Worst" or largest mean value for severity of concave portions of the contour',
    'concave_points_worst': '"Worst" or largest mean value for number of concave portions of the contour',
    'symmetry_worst': 'No description available',
    'fractal_dimension_worst': '"Worst" or largest mean value for "coastline approximation" - 1'
}

st.set_page_config(page_title="AI Medical Notes Screener - Breast Cancer", layout="wide")
st.title("AI Medical Notes Screener: Breast Cancer ⚕️")

st.header("File Upload")
st.write("An AI agent searches file for key parameters used in breast cancer diagnosis and a machine learning modelpredicts if the tumor is likely to be malignant or benign based on the extracted data.")
st.write("Upload structured or unstructured medical notes in PDF format for analysis.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
st.markdown("The content should include the ***mean***, ***worst value***, and ***standard error*** of the radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.")

if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.read()

        with pdfplumber.open(BytesIO(bytes_data)) as pdf:
            num_pages = len(pdf.pages)
            
            # Handle single vs multiple pages
            if num_pages == 1:
                selected_pages = (1, 1)
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    selected_pages = st.slider("Select pages to extract", 1, num_pages, (1, min(5, num_pages)))
            
            if st.button("Screen Notes for Breast Cancer Diagnosis"):
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

                # Make API call to Mistral AI Agent
                try:
                    api_key = os.environ.get("MISTRAL_API_KEY")
                    if not api_key:
                        st.error("Error: MISTRAL_API_KEY environment variable is not set. Please set it before running the app.")
                    else:
                        with st.spinner('Processing with AI Agent...'):
                            client = Mistral(api_key=api_key)
                            
                            inputs = [
                                {"role": "user", "content": full_text}
                            ]
                            
                            response = client.beta.conversations.start(
                                agent_id="ag_019d296fed7e7178806606b6787af470",
                                agent_version=6,
                                inputs=inputs,
                            )
                        
                        # Extract JSON from Mistral response
                        try:
                            # Get the content from the response
                            response_content = response.outputs[0].content
                            
                            # Extract JSON from markdown code block
                            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_content, re.DOTALL)
                            
                            if json_match:
                                json_str = json_match.group(1)
                                extracted_data = json.loads(json_str)
                                
                                # Display extracted parameters as a table
                                st.subheader("Parameters Extracted by AI Agent")
                                
                                # Create table data
                                table_data = []
                                for param, value in extracted_data.items():
                                    description = FEATURE_DESCRIPTIONS.get(param, 'No description available')
                                    table_data.append({
                                        'Parameter': param,
                                        'Value': value,
                                        'Description': description
                                    })
                                
                                # Create and display DataFrame
                                params_df = pd.DataFrame(table_data)
                                
                                # Use Streamlit's dataframe with custom styling
                                st.dataframe(
                                    params_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        'Parameter': st.column_config.TextColumn(width="medium"),
                                        'Value': st.column_config.TextColumn(width="small"),
                                        'Description': st.column_config.TextColumn(width="large")
                                    }
                                )
                                
                                # Map extracted features to model input (need all 30 features)
                                feature_names = [
                                    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                                    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
                                    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
                                    'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
                                    'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
                                    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                                    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
                                    'symmetry_worst', 'fractal_dimension_worst'
                                ]
                                
                                # Normalize extracted data: convert space-separated names to underscores
                                normalized_data = {}
                                for key, value in extracted_data.items():
                                    normalized_key = key.replace(' ', '_')
                                    normalized_data[normalized_key] = value
                                extracted_data = normalized_data
                                
                                # Check if all required parameters are present and valid
                                missing_parameters = []
                                invalid_parameters = []
                                
                                for feature in feature_names:
                                    if feature not in extracted_data:
                                        missing_parameters.append(feature)
                                    else:
                                        # Check if value is valid (not None, NaN, or non-numeric)
                                        value = extracted_data[feature]
                                        try:
                                            numeric_value = float(value)
                                            # Check if it's NaN
                                            if pd.isna(numeric_value):
                                                invalid_parameters.append((feature, "NaN value"))
                                        except (ValueError, TypeError):
                                            invalid_parameters.append((feature, f"non-numeric value: {value}"))
                                
                                if missing_parameters or invalid_parameters:
                                    # Display error if parameters are missing or invalid
                                    if missing_parameters:
                                        st.error(f"❌ Cannot run ML prediction - Missing {len(missing_parameters)} required parameter(s):")
                                        st.write("Missing parameters:")
                                        for param in missing_parameters:
                                            st.write(f"  • {param}")
                                    
                                    if invalid_parameters:
                                        st.error(f"❌ Invalid parameter values detected - {len(invalid_parameters)} parameter(s) with invalid data:")
                                        st.write("Invalid parameters:")
                                        for param, reason in invalid_parameters:
                                            st.write(f"  • {param}: {reason}")
                                    
                                    st.info("Please ensure the medical notes contain valid numeric measurements for all required features (mean, standard error, and worst values).")
                                else:
                                    # All parameters present and valid - proceed with prediction
                                    # Create feature vector with extracted values
                                    feature_vector = []
                                    for feature in feature_names:
                                        feature_vector.append(float(extracted_data[feature]))
                                    
                                    # Create DataFrame and predict
                                    input_df = pd.DataFrame([feature_vector], columns=feature_names)
                                    
                                    # Load model and scaler
                                    with open('breast_cancer_model.pkl', 'rb') as f:
                                        model = pickle.load(f)
                                    with open('scaler.pkl', 'rb') as f:
                                        scaler = pickle.load(f)
                                    
                                    # Scale and predict
                                    scaled_input = scaler.transform(input_df.values)
                                    prediction = model.predict(scaled_input)
                                    
                                    # Display prediction result with color coding
                                    st.subheader("ML Model Prediction Result")
                                    diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
                                    
                                    if diagnosis == "Benign":
                                        st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>Diagnosis: <span style='color: green;'>{diagnosis}</span> ✅</p>", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>Diagnosis: <span style='color: red;'>{diagnosis}</span> ⚠️</p>", unsafe_allow_html=True)
                                    
                                    st.info("*This is a machine learning prediction and should not be relied upon as a medical diagnosis. Always consult a healthcare professional for medical advice.*")
                            else:
                                st.warning("Could not extract JSON from Mistral response")
                                
                        except json.JSONDecodeError as json_error:
                            st.error(f"Error parsing JSON from Mistral response: {json_error}")
                        except Exception as ml_error:
                            st.error(f"Error processing prediction: {ml_error}")
                except Exception as ai_error:
                    st.error(f"Error calling Mistral AI: {ai_error}")

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
