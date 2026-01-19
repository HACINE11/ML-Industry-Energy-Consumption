from .rf import build_model as rf_build, param_grid as rf_grid
from .tree import build_model as tree_build, param_grid as tree_grid
from .linear import build_model as lin_build, param_grid as lin_grid


MODEL_REGISTRY = {
    "rf": (rf_build, rf_grid),
    "tree": (tree_build, tree_grid),
    "linear": (lin_build, lin_grid),

}