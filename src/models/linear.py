from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

MODEL_NAME = "linear"


def build_model(preprocessor):
    return Pipeline([
        ("pre", preprocessor),
        ("model", LinearRegression()),
    ])


param_grid = {}
