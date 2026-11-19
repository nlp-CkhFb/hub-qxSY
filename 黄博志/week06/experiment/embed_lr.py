"""
冻结特征 + 逻辑回归：不更新预训练权重，只把 [CLS] 当句向量。

这个实验回答的是：
  “如果不做微调、只把预训练模型当特征提取器，换骨干还有没有差别？”
它比微调便宜得多，也更接近实际业务里“先离线算向量、再轻量分类”的用法。
"""

import time

import numpy as np
from sklearn.linear_model import LogisticRegression

import config
import models


def _feature_paths(kind: str, pool: str):
    return (
        config.FEATURE_DIR / f"{kind}_{pool}_train.npy",
        config.FEATURE_DIR / f"{kind}_{pool}_test.npy",
    )


def run_frozen(kind: str, train, test, pool: str = "cls", force: bool = False):
    config.ensure_dirs()
    train_path, test_path = _feature_paths(kind, pool)

    y_train = np.array([r["label"] for r in train], dtype=np.int64)
    y_test = np.array([r["label"] for r in test], dtype=np.int64)

    extract_train = 0.0
    extract_test = 0.0
    if force or not (train_path.exists() and test_path.exists()):
        backbone = models.load_backbone(kind)
        tokenizer = models.load_tokenizer(str(config.BACKBONES[kind]["dir"]))

        t0 = time.perf_counter()
        X_train = models.encode_texts(
            backbone,
            tokenizer,
            [r["sentence"] for r in train],
            max_len=config.MAX_LEN,
            batch_size=64,
            pool=pool,
            normalize=True,
        )
        extract_train = time.perf_counter() - t0

        t1 = time.perf_counter()
        X_test = models.encode_texts(
            backbone,
            tokenizer,
            [r["sentence"] for r in test],
            max_len=config.MAX_LEN,
            batch_size=64,
            pool=pool,
            normalize=True,
        )
        extract_test = time.perf_counter() - t1

        np.save(train_path, X_train)
        np.save(test_path, X_test)
    else:
        X_train = np.load(train_path)
        X_test = np.load(test_path)

    # 训练逻辑回归
    t2 = time.perf_counter()
    clf = LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    lr_train = time.perf_counter() - t2

    # 推理 = 测试集编码（若缓存则为 0）+ LR 预测
    t3 = time.perf_counter()
    pred = clf.predict(X_test).astype(np.int64)
    prob = clf.predict_proba(X_test)
    lr_infer = time.perf_counter() - t3

    return {
        "pred": pred,
        "prob": prob,
        "train_seconds": extract_train + lr_train,
        "infer_seconds": extract_test + lr_infer,
        "extra": {
            "feature_dim": int(X_train.shape[1]),
            "pool": pool,
            "extract_train_seconds": round(extract_train, 3),
            "extract_test_seconds": round(extract_test, 3),
            "lr_train_seconds": round(lr_train, 3),
            "checkpoint_mb": config.BACKBONES[kind]["size_mb"],
        },
    }
