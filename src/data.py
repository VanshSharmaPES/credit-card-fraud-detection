"""Download, clean, split and scale the credit card fraud dataset.

Run once:      python src/data.py
In notebooks:  from data import load_splits
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

# Paths are built from the repo root so this works from notebooks/ too
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# If this link stops working, download the same file from Kaggle
# (dataset "mlg-ulb/creditcardfraud") and put it at data/raw/creditcard.csv
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# Expected shape of the raw file (before dropping duplicates)
RAW_ROWS = 284_807
RAW_COLS = 31
RAW_FRAUDS = 492

# The 30 model inputs. Raw Time is dropped and replaced by Hour.
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]
COLUMNS = ["row_id"] + FEATURES + ["Amount_raw", "Class"]


def download():
    """Fetch the raw CSV if it is not already on disk."""
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, RAW_PATH)
    return RAW_PATH


def load_raw():
    """Read the raw CSV and check it matches the known dataset."""
    df = pd.read_csv(download())
    assert df.shape == (RAW_ROWS, RAW_COLS), f"Unexpected shape {df.shape}"
    assert int(df["Class"].sum()) == RAW_FRAUDS, f"Unexpected fraud count {df['Class'].sum()}"
    return df


def clean(raw):
    """Drop duplicates and build the feature columns."""
    # Drop duplicates on the original 31 columns first. The index survives,
    # so row_id below is the original row number in creditcard.csv.
    df = raw.drop_duplicates().copy()
    n_dupes = len(raw) - len(df)

    df["row_id"] = df.index
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Amount_raw"] = df["Amount"]
    df["Amount"] = np.log1p(df["Amount"])
    df = df.drop(columns="Time")
    return df[COLUMNS].reset_index(drop=True), n_dupes


def split(df):
    """Stratified 60 / 20 / 20 split."""
    train, rest = train_test_split(df, test_size=0.4, stratify=df["Class"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["Class"], random_state=SEED)
    return train.copy(), val.copy(), test.copy()


def scale(train, val, test):
    """Fit the scaler on train only, then apply it to all three splits."""
    scaler = StandardScaler().fit(train[FEATURES])
    for part in (train, val, test):
        part[FEATURES] = scaler.transform(part[FEATURES])
    return train, val, test


def main():
    raw = load_raw()
    df, n_dupes = clean(raw)
    train, val, test = scale(*split(df))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    print(f"Raw rows: {len(raw)}, raw frauds: {int(raw['Class'].sum())}")
    print(f"Duplicates dropped: {n_dupes}")
    print(f"Rows after cleaning: {len(df)}, frauds: {int(df['Class'].sum())}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        frauds = int(part["Class"].sum())
        print(f"{name:<5} rows: {len(part):>7}  frauds: {frauds:>4}  fraud share: {frauds / len(part):.4%}")
    n_normal = int((train["Class"] == 0).sum())
    n_fraud = int(train["Class"].sum())
    print(f"Class weight (train normal / train fraud): {n_normal} / {n_fraud} = {n_normal / n_fraud:.4f}")


def load_splits():
    """Return X_train, y_train, X_val, y_val, X_test, y_test,
    rowid_val, rowid_test, amount_val, amount_test.

    X are DataFrames with the 30 features (already scaled). y, row ids and
    amounts are Series. amount_* is the raw amount in euros, only for the
    money analysis, never a model input.
    """
    paths = [PROCESSED_DIR / f"{name}.csv" for name in ("train", "val", "test")]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError("Processed splits not found. Run: python src/data.py")
    train, val, test = (pd.read_csv(p) for p in paths)
    return (
        train[FEATURES], train["Class"],
        val[FEATURES], val["Class"],
        test[FEATURES], test["Class"],
        val["row_id"], test["row_id"],
        val["Amount_raw"], test["Amount_raw"],
    )


if __name__ == "__main__":
    main()
