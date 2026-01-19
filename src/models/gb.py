from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor


def build_model(preprocessor):
    return Pipeline([
        ("pre", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42)),
    ])


param_grid = {
    "model__n_estimators": [100, 200, 500],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 4, 5],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}
