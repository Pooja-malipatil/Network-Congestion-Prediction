"""4G Network Congestion Classifier (simple version).

Steps:
1. Load data
2. Remove columns that don't help
3. Compare two models (Random Forest vs Gradient Boosting)
4. Use the better one to predict the test file
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42
TARGET = "Congestion_Type"
IDENTIFIER = "cell_name"

# Columns that are the same in every row or add nothing (found during analysis)
USELESS_COLUMNS = ["par_year", "par_month", "4G_rat"]


def load_csv(path):
    """Read a CSV file, with a clear message if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Keep the CSV files inside the data/ folder.")
    return pd.read_csv(path)


def make_pipeline(model, X):
    """Vendor name -> numbers (one-hot); all other columns are used as they are."""
    text_cols = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([("vendor", OneHotEncoder(handle_unknown="ignore"), text_cols)],
                             remainder="passthrough")
    return Pipeline([("prep", prep), ("model", model)])


def main():
    # 1. Load data
    train = load_csv(DATA_DIR / "train_upd.csv")
    test = load_csv(DATA_DIR / "test_upd.csv")

    # 2. Prepare data
    y = train[TARGET].astype(str)
    X = train.drop(columns=[TARGET, IDENTIFIER] + USELESS_COLUMNS, errors="ignore")
    X_test = test.drop(columns=[IDENTIFIER] + USELESS_COLUMNS, errors="ignore")
    X_test = X_test.reindex(columns=X.columns)
    print(f"Training rows: {len(train):,} | Features used: {X.shape[1]}")

    # 3. Two models to compare
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=18, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, random_state=RANDOM_STATE),
    }

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    results = {}
    for name, model in models.items():
        pipe = make_pipeline(model, X)
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_valid, pipe.predict(X_valid))
        # 5-fold cross-validation: a more reliable score than one split
        cv = cross_val_score(make_pipeline(model, X), X, y, cv=5, n_jobs=-1).mean()
        results[name] = (acc, cv, pipe)
        print(f"{name:18s} | validation accuracy: {acc:.4f} | 5-fold CV: {cv:.4f}")

    # 4. Pick the better model
    best_name = max(results, key=lambda n: results[n][1])
    best_pipe = results[best_name][2]
    print(f"\nBest model: {best_name}\n")

    pred = best_pipe.predict(X_valid)
    labels = sorted(y.unique())
    report = classification_report(y_valid, pred)
    matrix = confusion_matrix(y_valid, pred, labels=labels)
    print(report)
    print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())

    # 5. Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = "\n".join(f"{n}: validation={a:.4f}, cv={c:.4f}" for n, (a, c, _) in results.items())
    (OUTPUT_DIR / "validation_report.txt").write_text(
        f"{summary}\n\nBest model: {best_name}\n\n{report}\n{matrix}\n", encoding="utf-8")

    # Which inputs matter most? Shuffle one column at a time and see how much accuracy drops.
    sample = X_valid.sample(3000, random_state=RANDOM_STATE)
    imp = permutation_importance(best_pipe, sample, y_valid.loc[sample.index],
                                 n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", header=["importance"], index_label="feature")
    print("\nTop 5 important features:")
    print(importance.head(5).round(4).to_string())

    # Train the best model on ALL labelled data, then predict the test file
    final = make_pipeline(models[best_name], X).fit(X, y)
    out = pd.DataFrame({IDENTIFIER: test[IDENTIFIER], TARGET: final.predict(X_test)})
    out.to_csv(OUTPUT_DIR / "congestion_predictions.csv", index=False)
    joblib.dump(final, OUTPUT_DIR / "model.joblib")
    print("\nSaved in outputs/: congestion_predictions.csv, model.joblib, feature_importance.csv")


if __name__ == "__main__":
    main()