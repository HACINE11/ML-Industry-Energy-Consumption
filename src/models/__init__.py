# Import des modèles disponibles
from .rf import MODEL_NAME as RF_MODEL_NAME, build_model as build_rf_model, param_grid as rf_param_grid
from .knn import MODEL_NAME as KNN_MODEL_NAME, build_model as build_knn_model, param_grid as knn_param_grid

# Dictionnaire pour accéder facilement aux modèles
MODELS = {
    RF_MODEL_NAME: {
        'build': build_rf_model,
        'params': rf_param_grid
    },
    KNN_MODEL_NAME: {
        'build': build_knn_model,
        'params': knn_param_grid
    }
}
