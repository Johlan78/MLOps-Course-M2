#!/bin/bash

@echo "Creating Python virtual environment 'MLOps-env"; \
python3 -m venv MLOps-env; \
@echo "Activating Python virtual environment 'MLOps-env"; \
source MLOps-env/bin/activate; 
@echo "Installing requirements"; \
pip3 install -r requirements.txt; \
echo "Initialization complete"