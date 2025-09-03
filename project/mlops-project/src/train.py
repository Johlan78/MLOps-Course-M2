import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.pipeline import build_pipeline
from src.utils import load_config, load_data, split_data

# Configurez MLflow pour l'autologging
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

def train_model(config_path):
    config = load_config(config_path)

    # 1. Charger et préparer les données
    X, y = load_data(config['data']['csv_path'], config['data']['target'])
    X_train, X_test, y_train, y_test = split_data(X, y, 
                                                config['data']['test_size'],
                                                config['data']['random_state'])

    # 2. Créer le pipeline et la grille de paramètres
    pipeline = build_pipeline(config['features']['numeric'],
                            config['features']['categorical'],
                            config['model']['type'])

    param_grid = {f'model__{key}': value for key, value in config['model']['params'].items()}

    # 3. Utiliser GridSearchCV pour le tuning
    with mlflow.start_run(run_name="GridSearch_Training") as run:
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=config['cv']['n_splits'],
            scoring=config['cv']['scoring'],
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        # 4. Enregistrer les meilleurs paramètres et métriques
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_score", grid_search.best_score_)

        # Enregistrer le meilleur modèle et le mettre dans le registre
        mlflow.sklearn.log_model(grid_search.best_estimator_, 
                                artifact_path="model",
                                registered_model_name="ChurnClassifier")

        # 5. Évaluer le modèle sur l'ensemble de test
        score_test = grid_search.score(X_test, y_test)
        mlflow.log_metric("test_score", score_test)

        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print(f"Test set score: {score_test:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    train_model(args.config)
