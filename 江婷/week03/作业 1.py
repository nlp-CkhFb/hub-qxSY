# -*- coding: utf-8 -*-
import os
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===================== 全局超参配置 =====================
SEED = 42
MAX_LEN = 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
VOCAB_PATH = os.path.join(HERE, "data", "vocab.txt")

# --------------------------
# 标签控制开关：
# 设为 None = 自动从 train.csv 读取标签（日常默认，推荐）
# 填列表 = 强制白名单，例如 LABELS = ["ar", "en", "es", "hi", "ja", "pt", "ru", "zh", "ko", "th", "vi", "tr"]
# --------------------------
LABELS = None


# ===================== 文本预处理 & 词表工具 =====================
def normalize(text: str) -> str:
    """NFKC归一化、小写，只保留字母L和组合标记M"""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


def build_char_vocab(sentences: list[str], min_freq = 1):
    """构建字符词表：<PAD>=0，<UNK>=1，其余按频次降序"""
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(
        ((c, n) for c, n in freq.items() if n >= min_freq),
        key=lambda kv: (-kv[1], kv[0])
    )
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab(vocab: OrderedDict, freq: Counter, path: str):
    """保存字符表：字符\tid\t频次"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def load_vocab_file(path: str) -> OrderedDict:
    """读取字符表，跳过注释行"""
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ch, idx, _freq = line.split("\t")
            vocab[ch] = int(idx)
    return vocab


def encode(text: str, vocab: OrderedDict, maxlen: int = MAX_LEN) -> list[int]:
    """文本转id序列，截断+pad补0"""
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ===================== 数据加载 & Dataset =====================
def load_split(data_dir: str, name: str):
    """读取csv划分，返回 (text, label) 列表"""
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows: list[tuple[str, str]], vocab: OrderedDict, label2id: dict):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ===================== 模型定义 =====================
class LangIDClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # mask屏蔽padding位置，max池化不看pad
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.rnn(self.embedding(x))
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)


# ===================== 评估指标工具 =====================
def predict(model: nn.Module, loader: DataLoader):
    """批量预测，返回预测id列表、真实id列表"""
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            preds += logits.argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds: list[int], golds: list[int], num_classes: int):
    """宏F1：每类单独算F1再求平均"""
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def confusion(preds: list[int], golds: list[int], labels: list[str]):
    """构造混淆矩阵：行=真实标签，列=预测标签"""
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=3):
    """输出top高频混淆对（真实→预测）"""
    pairs = []
    for g in range(len(labels)):
        for p in range(len(labels)):
            if g != p:
                pairs.append((mat[g][p], labels[g], labels[p]))
    pairs.sort(reverse=True)
    print("  最常混淆（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt}")


# ===================== 训练一轮封装 =====================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss


# ===================== 主流程 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0, help="每类训练样本上限，0=不限制")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="模型保存路径")
    ap.add_argument("--make-vocab", action="store_true", help="强制重建字符表")
    args = ap.parse_args()

    # 环境&随机种子固定
    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. 加载数据划分
    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    train_label_set = {l for _, l in train_rows}
    global LABELS
    if LABELS is not None:
        # 手动指定白名单模式
        extra_labels = [l for l in train_label_set if l not in LABELS]
        if extra_labels:
            raise SystemExit(f"训练集存在白名单以外的标签：{' '.join(extra_labels)}")
    else:
        # 自动读取、排序去重，保证id稳定
        LABELS = sorted(train_label_set)
        print(f"✅ 自动从训练集读取标签：{LABELS}")

    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    # 校验 dev / test 不能出现训练集没有的标签
    dev_label_set = {l for _, l in dev_rows}
    test_label_set = {l for _, l in test_rows}
    all_unknown = (dev_label_set | test_label_set) - set(LABELS)
    if all_unknown:
        raise SystemExit(f"验证/测试集出现训练集不存在的标签：{' '.join(all_unknown)}")

    # 2. 每类采样控制（均衡采样）
    if args.per_class > 0:
        by_label = {}
        for text, lab in train_rows:
            by_label.setdefault(lab, []).append(text)
        train_rows = []
        rng = random.Random(SEED)
        for lab, texts in by_label.items():
            sample = rng.sample(texts, min(args.per_class, len(texts)))
            train_rows.extend([(t, lab) for t in sample])
        rng.shuffle(train_rows)

    # 3. 词表：新建 / 加载
    if args.make_vocab or not os.path.exists(args.vocab):
        print("构建字符表...")
        sentences = [text for text, _ in train_rows]
        vocab, freq = build_char_vocab(sentences)
        save_vocab(vocab, freq, args.vocab)
        print(f"字符表已保存至 {args.vocab}")
    else:
        vocab = load_vocab_file(args.vocab)
        print(f"加载已有字符表 {args.vocab}")

    # 基础信息打印
    print(f"训练 {len(train_rows)} / 验证 {len(dev_rows)} / 测试 {len(test_rows)}")
    print(f"类别数 {num_classes}，字符表大小 {len(vocab)}")
    print(f"随机猜测基准：{1/num_classes:.1%}")

    # 4. 构造DataLoader
    train_ds = TextDataset(train_rows, vocab, label2id)
    dev_ds = TextDataset(dev_rows, vocab, label2id)
    test_ds = TextDataset(test_rows, vocab, label2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    # 5. 模型、损失、优化器
    model = LangIDClassifier(len(vocab), num_classes)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 6. 训练循环
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        dev_preds, dev_golds = predict(model, dev_loader)
        dev_acc = sum(p == g for p, g in zip(dev_preds, dev_golds)) / len(dev_golds)
        print(f"Epoch {epoch:2d}/{args.epochs} loss={avg_loss:.4f} dev_acc={dev_acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒")

    # 7. 测试集全套评估
    test_preds, test_golds = predict(model, test_loader)
    test_acc = sum(p == g for p, g in zip(test_preds, test_golds)) / len(test_golds)
    test_f1 = macro_f1(test_preds, test_golds, num_classes)
    print(f"\n测试集 accuracy = {test_acc:.4f}   macro-F1 = {test_f1:.4f}")
    print_confusion(confusion(test_preds, test_golds, LABELS), LABELS)

    # 每一类单独准确率
    print("\n各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx_list = [i for i, g in enumerate(test_golds) if g == c]
        hit = sum(test_preds[i] == c for i in idx_list)
        total = len(idx_list)
        print(f"    {lab}: {hit}/{total} = {hit / max(1, total):.3f}")

    # 8. 推理示例展示
    print("\n--- 推理示例 ---")
    model.eval()
    demo_texts = [t for t, _ in test_rows[:4]]
    with torch.no_grad():
        for text in demo_texts:
            x = torch.tensor([encode(text, vocab)])
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    # 9. 保存模型（labels自动是读到的标签列表）
    if args.save:
        ckpt = {
            "state_dict": model.state_dict(),
            "vocab": dict(vocab),
            "labels": LABELS,
            "maxlen": MAX_LEN,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT
        }
        torch.save(ckpt, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
