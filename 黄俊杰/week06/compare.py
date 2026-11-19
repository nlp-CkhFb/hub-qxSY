# coding:utf8

# 解决 OpenMP 库冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import time
import random
import argparse
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

"""
TNEWS 新闻标题分类：四种方法对比
    规则 / TF-IDF + SVM / rbt3 冻结特征 + SVM / rbt3 微调
数据：data/train.json（3000 条）、data/val.json（500 条，已剔除与训练集重复的样本）
用法：
    python compare.py
    python compare.py --methods rule,tfidf
    python compare.py --smoke
"""

LABELS = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育",
          "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]
HERE = os.path.dirname(os.path.abspath(__file__))


def load_data(path):
    data = json.load(open(path, encoding="utf-8"))
    return [d["sentence"] for d in data], np.array([d["label"] for d in data])


def score(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro",
                             labels=list(range(len(LABELS))), zero_division=0),
    }


# 方法一：规则 / 关键词
def run_rule(train_texts, train_labels, val_texts, val_labels):
    fallback = Counter(train_labels).most_common(1)[0][0]
    preds, hit = [], 0
    for text in val_texts:
        pred = fallback
        for i, name in enumerate(LABELS):
            if name in text:
                pred = i
                hit += 1
                break
        preds.append(pred)
    m = score(val_labels, preds)
    print(f"  关键词命中率 {hit / len(val_texts):.2%}")
    return m


# 方法二：TF-IDF + LinearSVC
def run_tfidf(train_texts, train_labels, val_texts, val_labels):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2),
                          min_df=2, max_features=20000, sublinear_tf=True)
    x_train = vec.fit_transform(train_texts)
    x_val = vec.transform(val_texts)
    clf = LinearSVC(class_weight="balanced", max_iter=3000)
    clf.fit(x_train, train_labels)
    return score(val_labels, clf.predict(x_val))


# 方法三 / 四：预训练 rbt3
def run_frozen(model_name, train_texts, train_labels, val_texts, val_labels):
    import torch
    from transformers import AutoTokenizer, AutoModel

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    def encode(texts):
        feats = []
        for i in range(0, len(texts), 64):
            batch = tok(texts[i:i + 64], padding=True, truncation=True,
                        max_length=64, return_tensors="pt")
            with torch.no_grad():
                out = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            feats.append(((out * mask).sum(1) / mask.sum(1)).numpy())
        return np.concatenate(feats)

    x_train = encode(train_texts)
    x_val = encode(val_texts)
    clf = LinearSVC(class_weight="balanced", max_iter=3000)
    clf.fit(x_train, train_labels)
    return score(val_labels, clf.predict(x_val))


def run_finetune(model_name, train_texts, train_labels, val_texts, val_labels,
                 epochs=2, batch_size=16, lr=2e-5):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(model_name)

    class DS(Dataset):
        def __init__(self, texts, labels):
            enc = tok(texts, padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt")
            self.ids, self.mask = enc["input_ids"], enc["attention_mask"]
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "attention_mask": self.mask[i],
                    "labels": self.labels[i]}

    train_dl = DataLoader(DS(train_texts, train_labels), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(DS(val_texts, val_labels), batch_size=batch_size * 2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS))

    weight = compute_class_weight("balanced", classes=np.arange(len(LABELS)),
                                  y=train_labels)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_dl:
            opt.zero_grad()
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in val_dl:
                logits = model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
                preds.extend(logits.argmax(-1).tolist())
        m = score(val_labels, preds)
        print(f"  epoch {epoch}: loss={total / len(train_dl):.4f} "
              f"acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f}")
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.path.join(HERE, "data", "train.json"))
    parser.add_argument("--val", default=os.path.join(HERE, "data", "val.json"))
    parser.add_argument("--methods", default="rule,tfidf,frozen,finetune")
    parser.add_argument("--model", default="hfl/rbt3")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="只跑 100/50 条，检查流程")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_texts, train_labels = load_data(args.train)
    val_texts, val_labels = load_data(args.val)
    if args.smoke:
        train_texts, train_labels = train_texts[:100], train_labels[:100]
        val_texts, val_labels = val_texts[:50], val_labels[:50]
        args.epochs = 1

    print(f"train={len(train_texts)} val={len(val_texts)} model={args.model}")
    methods = {
        "rule": lambda: run_rule(train_texts, train_labels, val_texts, val_labels),
        "tfidf": lambda: run_tfidf(train_texts, train_labels, val_texts, val_labels),
        "frozen": lambda: run_frozen(args.model, train_texts, train_labels, val_texts, val_labels),
        "finetune": lambda: run_finetune(args.model, train_texts, train_labels,
                                         val_texts, val_labels, args.epochs,
                                         args.batch_size, args.lr),
    }

    results = []
    for name in [m.strip() for m in args.methods.split(",") if m.strip()]:
        print(f"\n[{name}]")
        t0 = time.time()
        m = methods[name]()
        m["method"] = name
        m["sec"] = time.time() - t0
        results.append(m)
        print(f"  acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"time={m['sec']:.1f}s")

    print("\n" + "=" * 56)
    print(f"{'method':<10} {'acc':>8} {'macro_f1':>10} {'time(s)':>9}")
    print("-" * 56)
    for m in results:
        print(f"{m['method']:<10} {m['acc']:>8.4f} {m['macro_f1']:>10.4f} {m['sec']:>9.1f}")

    with open(os.path.join(HERE, "results.csv"), "w", encoding="utf-8") as f:
        f.write("method,acc,macro_f1,sec\n")
        for m in results:
            f.write(f"{m['method']},{m['acc']:.4f},{m['macro_f1']:.4f},{m['sec']:.1f}\n")


if __name__ == "__main__":
    main()
