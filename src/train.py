from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocess import build_preprocessor
from src.models.registry import MODEL_REGISTRY


def chrono_split(df, target_col="Usage_kWh", test_ratio=0.2, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

    cut = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_train, y_train, X_test, y_test


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "Steel_industry_data.csv"
    df = pd.read_csv(data_path)

    target_col = "Usage_kWh"
    date_col = "date"

    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' introuvable. Colonnes: {list(df.columns)}")
    if date_col not in df.columns:
        raise ValueError(f"Colonne date '{date_col}' introuvable. Colonnes: {list(df.columns)}")

    X_train, y_train, X_test, y_test = chrono_split(df, target_col, 0.2, date_col)

    preprocessor = build_preprocessor(X_train)

    out_dir = root / "artifacts"
    out_dir.mkdir(exist_ok=True)

    cv = TimeSeriesSplit(n_splits=3)  # 3 folds = plus rapide (académique)

    results = []

    for name, (build_fn, grid) in MODEL_REGISTRY.items():
        print(f"\n==== {name} ====")
        pipe = build_fn(preprocessor)

        if grid:
            print("🔎 GridSearchCV...")
            search = GridSearchCV(
                pipe,
                param_grid=grid,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
        else:
            best_model = pipe.fit(X_train, y_train)
            best_params = {}

        preds = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)

        joblib.dump(best_model, out_dir / f"{name}.joblib")

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "best_params": str(best_params)
        })

        print("✅ best_params:", best_params)
        print(f"📊 MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

    res = pd.DataFrame(results).sort_values("RMSE")
    res.to_csv(out_dir / "model_results.csv", index=False)
    print(f"\n💾 Résultats: {out_dir / 'model_results.csv'}")


if __name__ == "__main__":
    main()
