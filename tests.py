"""Simple tests for the 4G Network Congestion project.
Run:  python tests.py      (run  python train_model.py  first)
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import joblib
import pandas as pd
from train_model import load_csv, IDENTIFIER, USELESS_COLUMNS, TARGET

ROOT = Path(__file__).resolve().parent
CLASSES = {"NC", "4G_RAN_CONGESTION", "4G_BACKHAUL_CONGESTION", "3G_BACKHAUL_CONGESTION"}
results = []


def check(test_id, name, func):
    try:
        message = func()
        results.append((test_id, name, "PASS", message))
    except Exception as e:                       # any error = fail
        results.append((test_id, name, "FAIL", f"{type(e).__name__}: {e}"))


def t1():
    df = load_csv(ROOT / "data" / "train_upd.csv")
    assert len(df) > 0
    return f"Loaded {len(df):,} rows, {df.shape[1]} columns"


def t2():
    model = joblib.load(ROOT / "outputs" / "model.joblib")
    df = load_csv(ROOT / "data" / "train_upd.csv")
    sample = df.groupby(TARGET).sample(200, random_state=1)
    X = sample.drop(columns=[TARGET, IDENTIFIER] + USELESS_COLUMNS)
    found = set(model.predict(X))
    assert found == CLASSES, f"classes found: {found}"
    return "All 4 classes predicted: " + ", ".join(sorted(found))


def t3():
    try:
        load_csv(ROOT / "data" / "missing_file.csv")
    except FileNotFoundError as e:
        return f"Error shown: {e}"
    raise AssertionError("No error was raised")


def t4():
    model = joblib.load(ROOT / "outputs" / "model.joblib")
    row = load_csv(ROOT / "data" / "test_upd.csv").drop(columns=[IDENTIFIER] + USELESS_COLUMNS).head(1)
    pred = model.predict(row)[0]
    assert pred in CLASSES
    return f"Predicted: {pred}"


def t5():
    path = ROOT / "outputs" / "model.joblib"
    assert path.exists(), "model.joblib not found"
    joblib.load(path)
    preds = pd.read_csv(ROOT / "outputs" / "congestion_predictions.csv")
    return f"model.joblib found ({path.stat().st_size/1e6:.1f} MB); predictions file has {len(preds):,} rows"


def t6():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
    assert not at.exception, at.exception[0].value
    return "Dashboard opened and showed a prediction: " + [m.value for m in at.metric][-1]


for tid, name, f in [("T1", "Dataset loading", t1), ("T2", "Congestion classification", t2),
                     ("T3", "Dataset validation (missing file)", t3), ("T4", "Model prediction", t4),
                     ("T5", "Model saving", t5), ("T6", "Dashboard prediction", t6)]:
    check(tid, name, f)

print()
for tid, name, status, msg in results:
    print(f"{tid} | {name:34s} | {status} | {msg}")
passed = sum(r[2] == "PASS" for r in results)
print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {len(results) - passed}")