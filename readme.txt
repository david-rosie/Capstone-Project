================================================================================
AI MEDICAL NOTES SCREENER (BREAST CANCER) - CAPSTONE PROJECT
================================================================================

PROJECT OVERVIEW
----------------
This is a project that uses generative AI and machine learning to quickly screen structured
or unstructured medical notes for potential breast cancer diagnosis.
The project includes:

1. Streamlit Front End Application - User-friendly interface
2. PDF Text Extractor - Extracts text from PDF documents
3. AI Agent - Extracts key parameters from medical notes text
4. Breast Cancer ML Model - A predictive model trained on wisconsin diagnostic dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

FEATURES
--------
- Web Interface: Interactive Streamlit application
- PDF Upload: Extracts text from PDF file (medical notes)
- Mistral AI Agent: Extracts key parameters from text and sends JSON to ML model
- Machine Learning Model: Predicts breast cancer classification using scikit-learn

PROJECT STRUCTURE
-----------------
- main.py                             - Main entry point
- Breast_Cancer_ML_Analysis.ipynb    - Jupyter notebook with ML analysis
- breast_cancer_ml_analysis.py        - Python ML analysis script
- frontend_app.py                     - Frontend application
- breast_cancer_model.pkl             - Trained ML model file
- scaler.pkl                          - Data scaler for preprocessing


REQUIREMENTS
------------
- streamlit
- pdfplumber
- kagglehub
- pandas
- seaborn
- scikit-learn
- mistralai
- python-dotenv


INSTALLATION
------------
1. Create a virtual environment:
   python -m venv .venv

2. Activate the virtual environment:
   - Windows: .venv\Scripts\activate
   - Linux/Mac: source .venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Create a .env file with Mistral AI API key "MISTRAL_API_KEY=..."


USAGE
-----
To run the Streamlit application:
   streamlit run frontend_app.py

To run the main application:
   python main.py


SETUP NOTES
-----------
- Ensure .env file is configured with Mistral API key
- Model and scaler files (.pkl) are pre-trained and ready to use

================================================================================
