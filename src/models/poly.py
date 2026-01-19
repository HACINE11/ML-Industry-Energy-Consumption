from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def build_model(preprocessor):
    return Pipeline([
        ("pre", preprocessor),
        ("poly", PolynomialFeatures(include_bias=False)),
        ("model", LinearRegression()),
    ])


param_grid = {
    "poly__degree": [2, 3],
}
