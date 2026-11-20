import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

LABELS = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育",
          "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]

GROUP_MAP = {
    "财经": "金融", "证券": "金融", "房产": "金融",
    "科技": "产业", "汽车": "产业", "农业": "产业",
    "国际": "宏观", "军事": "宏观",
    "教育": "噪音", "娱乐": "噪音", "体育": "噪音", "电竞": "噪音",
    "文化": "噪音", "故事": "噪音", "旅游": "噪音",
}
GROUPS = ["金融", "产业", "宏观", "噪音"]
RELEVANT = {"金融", "产业"}


def to_group(label_id):
    return GROUP_MAP[LABELS[label_id]]


def is_relevant(label_id):
    return to_group(label_id) in RELEVANT


def queue_name(label_id, conf, thr=0.8):
    if not is_relevant(label_id):
        return "drop"
    return "research" if conf >= thr else "review"


def load_splits():
    train = json.loads((DATA_DIR / "train.json").read_text(encoding="utf-8"))
    val = json.loads((DATA_DIR / "val.json").read_text(encoding="utf-8"))
    return train, val


def clean_val(train, val):
    seen = {d["sentence"] for d in train}
    clean = [d for d in val if d["sentence"] not in seen]
    return clean, len(val) - len(clean)


def sample(items, n, seed=42):
    if n <= 0 or n >= len(items):
        return list(items)
    return random.Random(seed).sample(items, n)


def load_eval(n=500, seed=42):
    train, val = load_splits()
    clean, n_dup = clean_val(train, val)
    evals = sample(clean, n, seed)
    meta = {
        "n_train": len(train),
        "n_val": len(val),
        "n_dup": n_dup,
        "n_clean": len(clean),
        "n_eval": len(evals),
        "seed": seed,
    }
    return evals, meta


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(y_true, y_pred, num_classes=len(LABELS)):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = y_pred >= 0
    recalls = {}
    for c in range(num_classes):
        mask = y_true == c
        recalls[LABELS[c]] = round(float((y_pred[mask] == c).mean()), 4) if mask.any() else None
    yt, yp = y_true[valid], y_pred[valid]
    macro = (f1_score(yt, yp, average="macro", labels=list(range(num_classes)),
                      zero_division=0) if valid.any() else 0.0)
    return {
        "n": int(len(y_true)),
        "unparsed": int((~valid).sum()),
        "acc": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(float(macro), 4),
        "per_class_recall": recalls,
        "confusion": (confusion_matrix(yt, yp, labels=list(range(num_classes))).tolist()
                      if valid.any() else []),
    }