from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

MODEL_NAME = "tree"

def build_model(preprocessor):
    return Pipeline([
        ("pre", preprocessor),
        ("model", DecisionTreeRegressor(random_state=42)),
    ])

param_grid = {
    "model__max_depth": [None, 10],
    "model__min_samples_split": [2, 10],
}

