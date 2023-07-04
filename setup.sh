#!/bin/bash

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables if necessary
# export VAR_NAME=value

# Run the Streamlit app
streamlit run app5.py