from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(numeric_features, categorical_features, model_type="logreg"):
    """
    Builds a machine learning pipeline with preprocessing and a specified model.
    
    Args:
        numeric_features (list): A list of column names for numeric features.
        categorical_features (list): A list of column names for categorical features.
        model_type (str): The type of model to use ('logreg' or 'random_forest').
        
    Returns:
        Pipeline: The complete scikit-learn pipeline.
    """
    
    # Define preprocessing steps for numeric features
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Define preprocessing steps for categorical features
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Combine the preprocessing steps using a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    # Select the model based on the input
    if model_type == "logreg":
        model = LogisticRegression(max_iter=500)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model_type. Choose 'logreg' or 'random_forest'.")
        
    # Combine the preprocessor and the model into a final pipeline
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
