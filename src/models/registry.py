"""
Registry pour tous les modèles du projet.
"""

MODEL_REGISTRY = {}

# Import des modèles existants
try:
    from .rf import build_model as rf_build, param_grid as rf_grid
    MODEL_REGISTRY["rf"] = (rf_build, rf_grid)
except ImportError:
    print("Note: rf.py non trouvé")

try:
    from .tree import build_model as tree_build, param_grid as tree_grid
    MODEL_REGISTRY["tree"] = (tree_build, tree_grid)
except ImportError:
    print("Note: tree.py non trouvé")

try:
    from .gb import build_model as gb_build, param_grid as gb_grid
    MODEL_REGISTRY["gb"] = (gb_build, gb_grid)
except ImportError:
    print("Note: gb.py non trouvé")

# Import de TON modèle KNN
from .knn import build_model as knn_build, param_grid as knn_grid
MODEL_REGISTRY["knn"] = (knn_build, knn_grid)

print(f"[registry] Modèles enregistrés: {list(MODEL_REGISTRY.keys())}")
