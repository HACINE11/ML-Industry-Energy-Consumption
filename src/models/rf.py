from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def build_model(preprocessor):
    return Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])


param_grid = {
    "model__n_estimators": [200, 500],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", "log2"],
}