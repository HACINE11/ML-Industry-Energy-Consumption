from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

MODEL_NAME = "knn"

def build_model(preprocessor):
    """
    Construit un pipeline KNN avec prétraitement
    """
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor())
    ])
    return model

# Grille de paramètres pour GridSearchCV
param_grid = {
    'regressor__n_neighbors': [3, 5, 7, 9, 11],
    'regressor__weights': ['uniform', 'distance'],
    'regressor__p': [1, 2],
    'regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'regressor__leaf_size': [10, 30, 50]
}
