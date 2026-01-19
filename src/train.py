from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# On suppose que ces fichiers existent dans ton projet
from src.preprocess import build_preprocessor  # à adapter si le nom est différent
from src.models.rf import build_model, param_grid


def main():
    root = Path(__file__).resolve().parents[1]  # .../src -> projet
    data_path = root / "Steel_industry_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {data_path}")

    df = pd.read_csv(data_path)

    # ⚠️ ADAPTE CES NOMS selon ton CSV
    target_col = "Usage_kWh"  # <-- mets ici le vrai nom de la colonne cible
    if target_col not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_col}' introuvable. Colonnes dispo: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor = build_preprocessor(X)

    pipe = build_model(preprocessor)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🔎 GridSearchCV en cours...")
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=3,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("✅ Meilleurs hyperparamètres :", search.best_params_)

    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"📊 MAE  = {mae:.4f}")
    print(f"📊 RMSE = {rmse:.4f}")
    print(f"📊 R2   = {r2:.4f}")

    out_dir = root / "artifacts"
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "rf_model.joblib"

    joblib.dump(best_model, model_path)
    print(f"💾 Modèle sauvegardé: {model_path}")


if __name__ == "__main__":
    main()
