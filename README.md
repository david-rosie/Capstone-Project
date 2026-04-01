# AI Medical Notes Screener (Breast Cancer) - Capstone Project

## Project Overview

This is a project that uses generative AI and machine learning to quickly screen structured or unstructured medical notes for potential breast cancer diagnosis. The project includes:

1. **Streamlit Front End Application** - User-friendly interface
2. **PDF Text Extractor** - Extracts text from PDF documents
3. **AI Agent** - Extracts key parameters from medical notes text
4. **Breast Cancer ML Model** - A predictive model trained on wisconsin diagnostic dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Features

- **Web Interface** - Interactive Streamlit application
- **PDF Upload** - Extracts text from PDF file (medical notes)
- **Mistral AI Agent** - Extracts key parameters from text and sends JSON to ML model
- **Machine Learning Model** - Predicts breast cancer classification using scikit-learn

## Project Structure

- `main.py` - Main entry point
- `Breast_Cancer_ML_Analysis.ipynb` - Jupyter notebook with ML analysis
- `breast_cancer_ml_analysis.py` - Python ML analysis script
- `frontend_app.py` - Frontend application
- `breast_cancer_model.pkl` - Trained ML model file
- `scaler.pkl` - Data scaler for preprocessing

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - **Windows:** `.venv\Scripts\activate`
   - **Linux/Mac:** `source .venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with Mistral AI API key:
   ```
   MISTRAL_API_KEY=...
   ```

## Usage

To run the Streamlit application:
```bash
streamlit run frontend_app.py
```

To run the main application:
```bash
python main.py
```

## Requirements

- streamlit
- pdfplumber
- kagglehub
- pandas
- seaborn
- scikit-learn
- mistralai
- python-dotenv

## Setup Notes

- Ensure `.env` file is configured with Mistral API key
- Model and scaler files (`.pkl`) are pre-trained and ready to use

---

*This is a capstone project for Digital Futures.*
