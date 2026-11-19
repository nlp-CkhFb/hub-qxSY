"""
数据准备：TNEWS 去重、固定切分。

为什么必须重新切分：
  - 原课程已经发现 TNEWS 验证集约 11% 与训练集重复，直接评测会虚高。
  - 本实验要比较两个骨干，必须保证它们看到完全相同的训练/验证/测试样本，
    否则模型差异和样本差异混在一起，结论不可信。

切分口径：
  - 训练集内部按 sentence 完全去重；
  - 验证集去掉与训练集重复的句子；
  - 从训练集分层抽 7,000 条训练、1,000 条开发集（用于选 epoch）；
  - 从去重后的验证集分层抽 2,000 条测试集（只用于最终报告）。
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import config


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _source_dir() -> Path:
    """优先用 week06/data 下的数据；没有就退回课程原始目录。"""
    if (config.DATA_DIR / "train.json").exists() and (config.DATA_DIR / "val.json").exists():
        return config.DATA_DIR
    if (config.COURSE_DATA_DIR / "train.json").exists():
        return config.COURSE_DATA_DIR
    raise FileNotFoundError(
        "找不到 TNEWS 数据。请把 train.json / val.json 放到 "
        f"{config.DATA_DIR}，或设置环境变量 TNEWS_DATA_DIR。"
    )


def _normalize(records, prefix: str):
    out = []
    for i, d in enumerate(records):
        sent = str(d["sentence"]).strip()
        if not sent:
            continue
        out.append(
            {
                "uid": f"{prefix}-{i}",
                "sentence": sent,
                "label": int(d["label"]),
            }
        )
    return out


def _stratified_sample(records, n: int, seed: int):
    """分层抽 n 条；n 必须小于等于 len(records)。"""
    if n >= len(records):
        return list(records)
    labels = [r["label"] for r in records]
    idx = np.arange(len(records))
    picked, _ = train_test_split(
        idx, train_size=n, random_state=seed, stratify=labels
    )
    return [records[i] for i in sorted(picked)]


def build_splits(force: bool = False):
    split_path = config.OUT_DIR / "splits.json"
    if split_path.exists() and not force:
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f)

    src = _source_dir()
    raw_train = _normalize(_read_json(src / "train.json"), "train")
    raw_val = _normalize(_read_json(src / "val.json"), "val")

    # 1) 训练集按整句去重（保留第一次出现）
    seen = set()
    train_clean = []
    for r in raw_train:
        if r["sentence"] in seen:
            continue
        seen.add(r["sentence"])
        train_clean.append(r)

    # 2) 验证集去掉与训练集重复的句子
    val_clean = [r for r in raw_val if r["sentence"] not in seen]

    # 3) 从训练集分层抽 8k，再切成 7k train / 1k dev
    labels_all = [r["label"] for r in train_clean]
    pool_idx, _ = train_test_split(
        np.arange(len(train_clean)),
        train_size=config.TRAIN_SIZE + config.DEV_SIZE,
        random_state=config.SEED,
        stratify=labels_all,
    )
    pool = [train_clean[i] for i in sorted(pool_idx)]

    dev = _stratified_sample(pool, config.DEV_SIZE, config.SEED + 1)
    dev_uids = {r["uid"] for r in dev}
    train_sub = _stratified_sample(
        [r for r in pool if r["uid"] not in dev_uids],
        config.TRAIN_SIZE,
        config.SEED + 2,
    )

    # 4) 从去重后的验证集分层抽 2k test
    test = _stratified_sample(val_clean, config.TEST_SIZE, config.SEED + 3)

    # 5) 断言三个集合互不重叠
    assert len({r["uid"] for r in train_sub}) == len(train_sub)
    assert len({r["uid"] for r in dev}) == len(dev)
    assert len({r["uid"] for r in test}) == len(test)
    assert not ({r["uid"] for r in train_sub} & {r["uid"] for r in dev})
    assert not ({r["uid"] for r in train_sub} & {r["uid"] for r in test})
    assert not ({r["uid"] for r in dev} & {r["uid"] for r in test})
    assert not (
        {r["sentence"] for r in train_sub}
        & {r["sentence"] for r in dev}
    )
    assert not (
        {r["sentence"] for r in train_sub}
        & {r["sentence"] for r in test}
    )

    def dist(records):
        arr = np.bincount([r["label"] for r in records], minlength=config.NUM_LABELS)
        return {config.LABELS[i]: int(arr[i]) for i in range(config.NUM_LABELS)}

    split = {
        "meta": {
            "source_dir": str(src),
            "seed": config.SEED,
            "raw_train": len(raw_train),
            "raw_val": len(raw_val),
            "train_dup_removed": len(raw_train) - len(train_clean),
            "val_dup_removed": len(raw_val) - len(val_clean),
            "train_clean": len(train_clean),
            "val_clean": len(val_clean),
            "train": len(train_sub),
            "dev": len(dev),
            "test": len(test),
            "max_len": config.MAX_LEN,
            "label_names": config.LABELS,
        },
        "train": train_sub,
        "dev": dev,
        "test": test,
        "label_dist": {
            "train": dist(train_sub),
            "dev": dist(dev),
            "test": dist(test),
        },
    }

    config.ensure_dirs()
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)
    return split


def load_splits(force: bool = False):
    split = build_splits(force=force)
    return split["train"], split["dev"], split["test"], split["meta"], split.get("label_dist")


if __name__ == "__main__":
    train, dev, test, meta, dist = load_splits(force=True)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\n训练集类别分布：")
    for k, v in dist["train"].items():
        print(f"  {k}: {v}")
    print("\n样例：")
    for r in train[:3] + test[:3]:
        print(f"  [{config.LABELS[r['label']]}] {r['sentence']}")
