from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def get_feature_groups(X):
    categorical_cols = ["Gender", "Position"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return categorical_cols, numeric_cols


def build_preprocessor(categorical_cols, numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )
    return preprocessor


# ALL MODELS USED

def build_random_forest():
    return RandomForestClassifier(n_estimators=200, random_state=42)

def build_logistic_regression():
    return LogisticRegression(max_iter=2000)

def build_gradient_boosting():
    return GradientBoostingClassifier()

def build_mlp():
    return MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1000)


def build_pipeline(preprocessor, model):
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    return pipeline