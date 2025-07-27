# Create virtual environment (choose one method)
python -m venv env       # Windows

# Activate environment
.\env\Scripts\activate   # Windows (PowerShell)

# Install the requirements
pip install -r requirements.txt

deactivate
rmdir env
pip cache purge

pip install -U langchain-community
pip install hf_xet





