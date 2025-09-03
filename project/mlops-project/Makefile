# ==============================================================================
# VARIABLES
# ==============================================================================
# La commande Python à utiliser. Assurez-vous que "python3" est le bon nom
# pour l'exécutable sur votre système. Si "python" fonctionne, utilisez-le.
PY=python3

# Le nom de l'environnement virtuel.
ENV?=.venv

# Le nom de l'expérimentation MLflow.
EXP?=churn-exp

# Le chemin vers le fichier de configuration par défaut.
CONFIG_FILE?=configs/config.yaml

# ==============================================================================
# CIBLES PRINCIPALES (TARGETS)
# ==============================================================================

.PHONY: all init train evaluate test clean

# Cible principale qui exécute le workflow complet : initialisation, entraînement, et évaluation.
all: init train evaluate

# Initialisation de l'environnement virtuel et installation des dépendances.
init:
	@echo "Creating Python virtual environment '$(ENV)'..."
	@$(PY) -m venv $(ENV)
	@echo "Activating virtual environment and installing dependencies from requirements.txt..."
	@. $(ENV)/bin/activate && pip install -U pip && pip install -r requirements.txt
	@echo "Initialization complete."

# Entraîne le modèle en utilisant le script src/train.py.
# Les paramètres sont définis via le fichier de configuration.
train:
	@echo "Starting model training and hyperparameter tuning..."
	MLFLOW_EXPERIMENT_NAME=$(EXP) . $(ENV)/bin/activate && PYTHONPATH=. $(PY) src/train.py --config $(CONFIG_FILE)
	@echo "Training complete. Check the MLflow UI for results."

# Évalue le modèle final et loggue les artefacts d'évaluation.
evaluate:
	@echo "Starting model evaluation and artifact logging..."
	MLFLOW_EXPERIMENT_NAME=$(EXP) . $(ENV)/bin/activate && PYTHONPATH=. $(PY) src/evaluate.py --config $(CONFIG_FILE)
	@echo "Evaluation complete. Artifacts have been logged."

# Exécute les tests unitaires pour vérifier la logique du pipeline.
test:
	@echo "Running tests with pytest..."
	@. $(ENV)/bin/activate && pytest -q tests/test_pipeline.py
	@echo "Tests complete."

# Nettoie les fichiers et dossiers générés par le projet pour repartir de zéro.
clean:
	@echo "Cleaning up generated files and directories..."
	@rm -rf $(ENV)
	@rm -rf data/processed/
	@rm -rf mlruns/
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "Cleanup complete."
