# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import random
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

"""

基于pytorch框架编写模型训练
实现一个语种识别任务（12分类）
规律：一句话按字符切分后送进RNN，模型输出它属于哪种语言
12类：ar en es hi ja pt ru zh ko th vi tr

"""

SEED = 42
MAX_LEN = 40  # 一句话最多取40个字符，不够的补0
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1  # 小模型单线程比多线程快

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

LABELS = ["ar", "en", "es", "hi", "ja", "pt", "ru", "zh", "ko", "th", "vi", "tr"]


# 归一化：NFKC + 转小写 + 只保留字母和组合符号（印地语、泰语的元音是组合符号）
def normalize(text):
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# 字符表：统计训练集里每个字符出现的次数，按次数从高到低编号（0号<PAD>补齐，1号<UNK>没见过）
def build_char_vocab(sentences):
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        vocab[ch] = len(vocab)
    return vocab, freq


# 12类的字符表课程没有提供，按课堂代码自己生成一份存下来
def save_vocab(vocab, freq, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


def load_split(name):
    rows = []
    with open(os.path.join(DATA_DIR, f"{name}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


class LangIDClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(LangIDClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)  # 0号补齐位不参与训练
        self.rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)  # 补齐位不参与池化，否则等于让模型读一串0
        h, _ = self.rnn(self.embedding(x))
        h = h.masked_fill(~mask, -1e9)
        pooled = self.dropout(self.norm(h.max(dim=1)[0]))
        return self.fc(pooled)  # 每行是12个语种的打分，softmax在loss里


# 测试代码，返回准确率、macro-F1和预测结果
def evaluate(model, loader, labels):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += model(x).argmax(dim=1).tolist()
            golds += y.tolist()
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1s = []
    for c in range(len(labels)):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return acc, sum(f1s) / len(f1s), preds, golds


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(NUM_THREADS)

    train_rows = load_split("train")
    dev_rows = load_split("dev")
    test_rows = load_split("test")
    label2id = {l: i for i, l in enumerate(LABELS)}

    vocab, freq = build_char_vocab([t for t, _ in train_rows])
    save_vocab(vocab, freq, VOCAB_PATH)
    print("训练 %d 条 / 验证 %d 条 / 测试 %d 条"
          % (len(train_rows), len(dev_rows), len(test_rows)))
    print("字符表 %d 个字符，已保存到 data/langid/vocab/vocab_v12.txt" % len(vocab))
    print("随机猜的准确率基准：%.1f%%" % (100 / len(LABELS)))

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), len(LABELS))
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    log = []
    for epoch in range(EPOCHS):
        model.train()
        watch_loss = []
        for x, y in train_loader:
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        mean_loss = sum(watch_loss) / len(watch_loss)
        acc = evaluate(model, dev_loader, LABELS)[0]
        print("第%d轮 loss:%f dev_acc:%f" % (epoch + 1, mean_loss, acc))
        log.append([acc, mean_loss])

    acc, f1, preds, golds = evaluate(model, test_loader, LABELS)
    print("\n测试集 accuracy:%f macro-F1:%f" % (acc, f1))
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        print("  %s: %d/%d" % (lab, sum(preds[i] == c for i in idx), len(idx)))
    pairs = Counter((LABELS[g], LABELS[p]) for p, g in zip(preds, golds) if p != g)
    print("最常混淆：%s" % pairs.most_common(3))

    torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                "labels": LABELS, "maxlen": MAX_LEN}, "langid_rnn_v12.pt")

    plt.plot(range(1, EPOCHS + 1), [l[0] for l in log], label="acc")
    plt.plot(range(1, EPOCHS + 1), [l[1] for l in log], label="loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("train-curve.png", dpi=200)
    plt.show()


# 使用训练好的模型做预测
def predict(model_path, texts):
    ckpt = torch.load(model_path)
    vocab = ckpt["vocab"]
    model = LangIDClassifier(len(vocab), len(ckpt["labels"]))
    model.load_state_dict(ckpt["state_dict"])

    model.eval()
    with torch.no_grad():
        logits = model(torch.LongTensor([encode(t, vocab) for t in texts]))
        prob = torch.softmax(logits, dim=1)
    for i, p in enumerate(prob):
        pred = int(torch.argmax(p))
        print("输入：%s, 预测语种：%s, 概率值：%f"
              % (texts[i], ckpt["labels"][pred], p[pred]))


if __name__ == "__main__":
    main()

    test_texts = ["今天天气很好", "今日はいい天気です", "The weather is nice today",
                  "El tiempo es bueno hoy", "Сегодня хорошая погода", "오늘 날씨가 좋아요"]
    predict("langid_rnn_v12.pt", test_texts)
