import shutil
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset
from config import SEED, TOP_N_CLASSES, WINDOW_SIZE, WINDOW_STRIDE, ALIGN_TOLERANCE

UCI_URL = "https://archive.ics.uci.edu/static/public/196/localization%2Bdata%2Bfor%2Bperson%2Bactivity.zip"
DATA_DIR = Path("posture_activity_data")
ZIP_PATH = DATA_DIR / "localization.zip"
RAW_PATH = DATA_DIR / "ConfLongDemo_JSI.txt"

TAG_MAP = {
    "010-000-024-033": "ankle_left",
    "010-000-030-096": "ankle_right",
    "020-000-032-221": "belt",
    "020-000-033-111": "chest",
}

RAW_COLUMNS = ["sequence", "tag", "timestamp", "datetime", "x", "y", "z", "activity"]
FEATURE_COLS = [
    "ankle_left_x", "ankle_left_y", "ankle_left_z",
    "ankle_right_x", "ankle_right_y", "ankle_right_z",
    "belt_x", "belt_y", "belt_z",
    "chest_x", "chest_y", "chest_z",
]


def load_raw_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        urllib.request.urlretrieve(UCI_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            name = next(n for n in zf.namelist() if n.endswith("ConfLongDemo_JSI.txt"))
            with zf.open(name) as src, open(RAW_PATH, "wb") as dst:
                shutil.copyfileobj(src, dst)

    raw = pd.read_csv(RAW_PATH, header=None, names=RAW_COLUMNS)
    raw["sequence"] = raw["sequence"].astype(str).str.strip()
    raw["tag"] = raw["tag"].astype(str).str.strip()
    raw["activity"] = raw["activity"].astype(str).str.strip().str.strip("'\"")

    for col in ["x", "y", "z"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    dt = raw["datetime"].astype(str).str.strip().str.replace(r":(\d{1,6})$", r".\1", regex=True)
    raw["dt"] = pd.to_datetime(dt, format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")
    raw["tag_name"] = raw["tag"].map(TAG_MAP)

    return raw.dropna(subset=["sequence", "activity", "dt", "x", "y", "z", "tag_name"]).sort_values(["sequence", "dt"]).reset_index(drop=True)


def synchronize_tags(raw):
    parts = []
    for sequence, group in raw.groupby("sequence"):
        group = group.sort_values("dt")
        merged = group[group["tag_name"] == "chest"][["dt", "activity", "x", "y", "z"]].rename(
            columns={"x": "chest_x", "y": "chest_y", "z": "chest_z"}
        )
        merged["sequence"] = sequence

        for tag in ["ankle_left", "ankle_right", "belt"]:
            other = group[group["tag_name"] == tag][["dt", "activity", "x", "y", "z"]].rename(
                columns={"x": f"{tag}_x", "y": f"{tag}_y", "z": f"{tag}_z"}
            )
            merged = pd.merge_asof(
                merged.sort_values("dt"),
                other.sort_values("dt"),
                on="dt",
                by="activity",
                direction="nearest",
                tolerance=pd.Timedelta(ALIGN_TOLERANCE),
            )
        parts.append(merged)

    return pd.concat(parts, ignore_index=True).dropna(subset=FEATURE_COLS).reset_index(drop=True)


def build_windows(df):
    X, y = [], []
    for _, sequence in df.groupby("sequence"):
        sequence = sequence.sort_values("dt").copy()
        sequence["segment"] = sequence["activity"].ne(sequence["activity"].shift()).cumsum()

        for _, segment in sequence.groupby("segment"):
            values = segment[FEATURE_COLS].to_numpy(dtype=np.float32)
            label = segment["activity"].iloc[0]
            for start in range(0, len(values) - WINDOW_SIZE + 1, WINDOW_STRIDE):
                X.append(values[start:start + WINDOW_SIZE].reshape(-1))
                y.append(label)

    return np.asarray(X, dtype=np.float32), np.asarray(y)


def prepare_data(X, y):
    X_train, X_test, y_train_text, y_test_text = train_test_split(
        X, y, test_size=0.5, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    encoder = LabelEncoder().fit(y)
    y_train = encoder.transform(y_train_text).astype(np.int64)
    y_test = encoder.transform(y_test_text).astype(np.int64)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    return SimpleNamespace(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        class_names=encoder.classes_.tolist(),
        scaler=scaler,
        label_encoder=encoder,
    )


def load_data():
    raw = load_raw_data()
    aligned = synchronize_tags(raw)
    top_classes = aligned["activity"].value_counts().head(TOP_N_CLASSES).index
    aligned = aligned[aligned["activity"].isin(top_classes)].sort_values(["sequence", "dt"]).reset_index(drop=True)
    X, y = build_windows(aligned)
    return prepare_data(X, y), pd.Series(y).value_counts()
