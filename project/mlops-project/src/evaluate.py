import argparse
import mlflow
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from src.utils import load_config, load_data, split_data
import matplotlib.pyplot as plt
import seaborn as sns

# Ajoutez des fonctions utilitaires pour les graphiques dans utils.py ou ici
def plot_and_log_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

def plot_and_log_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()

def evaluate_model(config_path):
    config = load_config(config_path)

    # 1. Charger les données de test
    X, y = load_data(config['data']['csv_path'], config['data']['target'])
    _, X_test, _, y_test = split_data(X, y, 
                                    config['data']['test_size'],
                                    config['data']['random_state'])

    # 2. Charger le modèle depuis le registre MLflow
    logged_model = 'runs:/<RUN_ID>/model' # Remplacez <RUN_ID> par l'ID du run d'entraînement
    model = mlflow.sklearn.load_model(logged_model)

    # 3. Faire des prédictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    with mlflow.start_run(run_name="Evaluation", nested=True):
        # 4. Créer et logger les artefacts
        plot_and_log_confusion_matrix(y_test, y_pred)
        plot_and_log_roc_curve(y_test, y_probs)

        # 5. Enregistrer les métriques d'évaluation
        # ... (Ajouter d'autres métriques si besoin)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    # Pour le moment, l'ID de run est codé en dur, à changer pour une solution plus robuste
    # comme lire le dernier run depuis MLflow.
    # Alternativement, on peut passer l'ID de run en paramètre
    # parser.add_argument("--run-id", type=str, required=True)
    # evaluate_model(args.config, args.run_id)

    evaluate_model(args.config)
