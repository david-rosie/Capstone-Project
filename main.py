#!/usr/bin/env python3
"""
Main entry point for the Breast Cancer ML Analysis Streamlit Application
"""

import subprocess
import sys
import os


def main():
    """Run the ML analysis and then start the Streamlit application"""
    
    print("=" * 60)
    print("Starting Breast Cancer ML Analysis Frontend")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ml_analysis_script = os.path.join(script_dir, "breast_cancer_ml_analysis.py")
    frontend_app = os.path.join(script_dir, "frontend_app.py")
    
    # Check if ML analysis script exists
    if not os.path.exists(ml_analysis_script):
        print(f"Error: breast_cancer_ml_analysis.py not found at {ml_analysis_script}")
        sys.exit(1)
    
    # Check if frontend_app.py exists
    if not os.path.exists(frontend_app):
        print(f"Error: frontend_app.py not found at {frontend_app}")
        sys.exit(1)
    
    # Run the ML analysis script first
    print("\n" + "=" * 60)
    print("Running ML Analysis...")
    print("=" * 60)
    try:
        result = subprocess.run(
            [sys.executable, ml_analysis_script],
            check=False
        )
        if result.returncode != 0:
            print(f"Warning: ML analysis script exited with code {result.returncode}")
    except Exception as e:
        print(f"Error running ML analysis script: {e}")
        sys.exit(1)
    
    # Run the frontend app
    print("\n" + "=" * 60)
    print("Starting Frontend Application...")
    print("=" * 60)
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", frontend_app],
            check=False
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Application stopped by user")
        print("=" * 60)
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
