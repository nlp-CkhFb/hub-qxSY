# coding:utf8

# 解决 OpenMP 库冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import time
import random
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

"""
DuEE-Fin 事件触发词抽取：规则 / BiLSTM / rbt3 三种方法
    输入一句公告，输出触发词 span + 事件类型
    数据：data/train.jsonl（1500 句）、data/dev.jsonl（400 句，触发词都在前 192 字内）
用法：
    python compare.py
    python compare.py --methods rule,bilstm
    python compare.py --smoke
"""

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.load(open(os.path.join(HERE, "data", "event_schema.json"), encoding="utf-8"))
TYPES = sorted(k for k in SCHEMA if not k.startswith("__"))
TAGS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}


def load_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def gold_span(row):
    e = row["event"]
    t = e["trigger"]
    return e["type"], t["start"], t["end"]


def make_tags(row, max_len):
    text = row["text"][:max_len]
    tags = ["O"] * len(text)
    typ, s, e = gold_span(row)
    if s >= len(text) or e > len(text):
        return tags
    tags[s] = f"B-{typ}"
    for i in range(s + 1, e):
        tags[i] = f"I-{typ}"
    return tags


def decode(tags):
    spans, cur = [], None
    for i, t in enumerate(tags):
        if t == "O":
            if cur:
                spans.append(cur); cur = None
        elif t.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = [t[2:], i, i + 1]
        elif t.startswith("I-"):
            if cur and cur[0] == t[2:]:
                cur[2] = i + 1
            else:
                if cur:
                    spans.append(cur)
                cur = [t[2:], i, i + 1]
    if cur:
        spans.append(cur)
    if not spans:
        return None
    return max(spans, key=lambda x: x[2] - x[1])


def evaluate(rows, preds):
    tp = fp = fn = type_ok = 0
    for row, pred in zip(rows, preds):
        g = gold_span(row)
        if pred is not None and tuple(pred) == g:
            tp += 1
        else:
            if pred is not None:
                fp += 1
            fn += 1
        if pred is not None and pred[0] == g[0]:
            type_ok += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "type_acc": type_ok / len(rows)}


def build_lex(train_rows):
    lex = defaultdict(Counter)
    for row in train_rows:
        e = row["event"]
        lex[e["trigger"]["text"]][e["type"]] += 1
    return lex


# 方法一：规则 / 触发词词典
def run_rule(train_rows, dev_rows, max_len):
    lex = build_lex(train_rows)
    triggers = sorted(lex, key=len, reverse=True)
    preds = []
    for row in dev_rows:
        text = row["text"][:max_len]
        best = None
        for trig in triggers:
            start = text.find(trig)
            while start >= 0:
                cand = (len(trig), -start, trig, start, start + len(trig))
                if best is None or cand > best:
                    best = cand
                start = text.find(trig, start + 1)
        if best:
            preds.append((lex[best[2]].most_common(1)[0][0], best[3], best[4]))
        else:
            preds.append(None)
    return preds


# 方法二：字符级 BiLSTM
class TagDS(Dataset):
    def __init__(self, rows, char2idx, max_len):
        self.items = []
        for row in rows:
            text = row["text"][:max_len]
            ids = [char2idx.get(c, 1) for c in text]
            labels = [TAG2ID[t] for t in make_tags(row, max_len)]
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    L = max(len(x) for x, _ in batch)
    ids = torch.zeros(len(batch), L, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
    return ids, labels


def tag_weights(train_rows, max_len):
    cnt = Counter(t for row in train_rows for t in make_tags(row, max_len))
    total = sum(cnt.values())
    w = [total / (len(TAGS) * max(cnt.get(t, 0), 1)) for t in TAGS]
    return torch.tensor([min(x, 20.0) for x in w], dtype=torch.float)


class BiLSTM(nn.Module):
    def __init__(self, vocab, n_tags, emb=128, hidden=128, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_tags)

    def forward(self, ids):
        x = self.drop(self.emb(ids))
        x, _ = self.lstm(x)
        return self.fc(self.drop(x))


@torch.no_grad()
def bilstm_predict(model, loader, rows, max_len):
    model.eval()
    preds = []
    for ids, _ in loader:
        tags = model(ids).argmax(-1).tolist()
        for row, seq in zip(rows[len(preds):], tags):
            text = row["text"][:max_len]
            preds.append(decode([ID2TAG[t] for t in seq[:len(text)]]))
    return preds


def run_bilstm(train_rows, dev_rows, max_len, epochs=5, batch_size=32, lr=1e-3):
    char2idx = {"<pad>": 0, "<unk>": 1}
    for row in train_rows:
        for c in row["text"][:max_len]:
            char2idx.setdefault(c, len(char2idx))
    train_dl = DataLoader(TagDS(train_rows, char2idx, max_len), batch_size=batch_size,
                          shuffle=True, collate_fn=collate, drop_last=True)
    dev_dl = DataLoader(TagDS(dev_rows, char2idx, max_len), batch_size=batch_size,
                        shuffle=False, collate_fn=collate)
    model = BiLSTM(len(char2idx), len(TAGS))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=tag_weights(train_rows, max_len), ignore_index=-100)

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for ids, labels in train_dl:
            loss = loss_fn(model(ids).reshape(-1, len(TAGS)), labels.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()
        preds = bilstm_predict(model, dev_dl, dev_rows, max_len)
        m = evaluate(dev_rows, preds)
        print(f"  epoch {epoch}: loss={total / len(train_dl):.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best


# 方法三：rbt3 冻结特征 + LinearSVC 判类型，再用该类型词典定位触发词
def run_bert_pipeline(train_rows, dev_rows, max_len, model_name):
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    def encode(rows):
        feats = []
        for i in range(0, len(rows), 64):
            texts = [r["text"][:max_len] for r in rows[i:i + 64]]
            enc = tok(texts, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt")
            with torch.no_grad():
                out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            feats.append(((out * mask).sum(1) / mask.sum(1)).numpy())
        return np.concatenate(feats)

    x_train = encode(train_rows)
    x_dev = encode(dev_rows)
    clf = LinearSVC(class_weight="balanced", max_iter=3000)
    clf.fit(x_train, [r["event"]["type"] for r in train_rows])
    type_pred = clf.predict(x_dev)
    clf_acc = accuracy_score([r["event"]["type"] for r in dev_rows], type_pred)

    # 类型由 rbt3 判，触发词 span 由规则词典定位；两者拼成最终事件
    rule_preds = run_rule(train_rows, dev_rows, max_len)
    preds = []
    for typ, rp in zip(type_pred, rule_preds):
        preds.append((typ, rp[1], rp[2]) if rp else None)
    return preds, clf_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.path.join(HERE, "data", "train.jsonl"))
    parser.add_argument("--dev", default=os.path.join(HERE, "data", "dev.jsonl"))
    parser.add_argument("--methods", default="rule,bilstm,bert")
    parser.add_argument("--model", default="hfl/rbt3")
    parser.add_argument("--max_len", type=int, default=192)
    parser.add_argument("--epochs_bilstm", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_rows = [r for r in load_rows(args.train)
                  if r["event"]["trigger"]["end"] <= args.max_len]
    dev_rows = [r for r in load_rows(args.dev)
                if r["event"]["trigger"]["end"] <= args.max_len]
    if args.smoke:
        train_rows, dev_rows = train_rows[:80], dev_rows[:40]
        args.epochs_bilstm = 1
    print(f"train={len(train_rows)} dev={len(dev_rows)} types={len(TYPES)}")

    results = []
    for name in [m.strip() for m in args.methods.split(",") if m.strip()]:
        print(f"\n[{name}]")
        t0 = time.time()
        if name == "rule":
            m = evaluate(dev_rows, run_rule(train_rows, dev_rows, args.max_len))
        elif name == "bilstm":
            m = run_bilstm(train_rows, dev_rows, args.max_len, args.epochs_bilstm,
                           args.batch_size)
        elif name == "bert":
            preds, clf_acc = run_bert_pipeline(train_rows, dev_rows, args.max_len, args.model)
            m = evaluate(dev_rows, preds)
            print(f"  type classifier acc={clf_acc:.4f}")
        else:
            raise ValueError(name)
        m["method"] = name
        m["sec"] = time.time() - t0
        results.append(m)
        print(f"  P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f} type_acc={m['type_acc']:.4f} time={m['sec']:.1f}s")

    print("\n" + "=" * 70)
    print(f"{'method':<8} {'P':>8} {'R':>8} {'F1':>8} {'type_acc':>9} {'time(s)':>9}")
    print("-" * 70)
    for m in results:
        print(f"{m['method']:<8} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['type_acc']:>9.4f} {m['sec']:>9.1f}")

    with open(os.path.join(HERE, "results.csv"), "w", encoding="utf-8") as f:
        f.write("method,precision,recall,f1,type_acc,sec\n")
        for m in results:
            f.write(f"{m['method']},{m['precision']:.4f},{m['recall']:.4f},"
                    f"{m['f1']:.4f},{m['type_acc']:.4f},{m['sec']:.1f}\n")


if __name__ == "__main__":
    main()
