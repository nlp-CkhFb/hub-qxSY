import torch
import torch.nn as nn
import csv
import os
import unicodedata
from collections import Counter, OrderedDict

from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED = 42
MAX_LEN = 40          # 中文中位长度约 10 字符、德语约 39，统一截到 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1       # 小模型单线程比 16 线程快 2.7 倍（实测），别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v8")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v8.txt")

# 课堂用的 8 个语种：模型的输出就是这 8 个格子
LABELS = ["ar", "en", "es", "hi", "ja", "pt", "ru", "zh", "ko", "vi", "tr", "th"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class LanguageClassifyModel(nn.Module):
    """语言识别模型"""
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(10000, 128, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=128,
            batch_first=True
        )
        self.linear = nn.Linear(128, 12)
        self.loss_fn = nn.CrossEntropyLoss()
        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.3)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x, y= None):
        output, h_n = self.rnn(self.embedding(x))
        mask = (x != 0).unsqueeze(-1)   
        output = output.masked_fill(~mask, -1e9)
        pooled = output.max(dim=1)[0]                   # (B, H)
        pooled = self.dropout(self.norm(pooled))
        return self.linear(pooled)

class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id.get(l, label2id["zh"]) for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]

def load_input_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        result = []
        for row in reader:
            result.append((row['text'], row['label']))
    return result

def build_vocab_from_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        sentences = []
        for row in reader:
            sentences.append(row['text'])
    build_vocab_from_sentences(sentences)
            
    
def build_vocab_from_sentences(sentences):
    freq = Counter()
    for s in sentences:
        print("sentence is ", s)
        normalized = normalize(s)
        print("normalized sentence is ", s)
        freq.update(normalized)
    
    with open("/Users/huangchengcheng/vscode/nlp-study-1/黄成成/week03/data/langid/vocab/vocab_v12.csv", mode="w") as f:
        fieldnames = ["char", "id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{"char": "<PAD>", "id": 0}, {"char": "<UNK>", "id": 1}])
        for i, (e, c) in enumerate(freq.items()):
            writer.writerow({"char": e, "id": i+2})

def encode(sentence, vocab):
    ids = [vocab.get(c, vocab["<UNK>"]) for c in normalize(sentence)][:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids

def load_vocab(file_path):
    vocab = {}
    with open(file_path, mode="r") as f:   
        reader = csv.DictReader(f)
        for row in reader:
            vocab[row['char']] = int(row['id'])
    return vocab

    
# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M：印地语、泰语的元音是组合标记（कि 里的 ि），
    只留字母会把印地语砍成裸辅音，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))

# ========== 单条预测函数 ==========
def predict_one(text: str, model, vocab, id2label, device="cpu"):
    model.eval()
    with torch.no_grad():
        # 文本转id
        ids = encode(text, vocab)
        # 构造tensor，增加batch维度 [1, seq_len]
        x = torch.tensor([ids], dtype=torch.long).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        pred_label = id2label[pred_idx]
        prob = probs[0][pred_idx].item()
    return pred_label, prob

# ─── 5. 评测 ───────────────────────────────────────────────────────────
def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds

def macro_f1(preds, golds, num_classes):
    """每一类各自算 F1，再对类别取平均（少数类不会被大类淹掉）。"""
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)

def main():
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for i, l in enumerate(LABELS)}
    print(label2id)
    vocab = load_vocab("/Users/huangchengcheng/vscode/nlp-study-1/黄成成/week03/data/langid/vocab/vocab_v12.csv");
    train_data = load_input_data(os.path.join(BASE_DIR, "data/langid/v12/train.csv"))
    test_data = load_input_data(os.path.join(BASE_DIR, "data/langid/v12/test.csv"))
    dev_data = load_input_data(os.path.join(BASE_DIR, "data/langid/v12/dev.csv"))
    train_loader = DataLoader(TextDataset(train_data, vocab, label2id), batch_size= 64, shuffle=True)
    test_loader = DataLoader(TextDataset(test_data, vocab, label2id), batch_size= 64, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_data, vocab, label2id), batch_size= 64, shuffle=True)

    model = LanguageClassifyModel()

    for epoch in range(1, EPOCHS+1):
        model.train()
        total = 0
        for X, y in train_loader:
            logits = model(X)
            loss = model.loss_fn(logits, y)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            total += loss.item()
        predictions, goals = predict(model, dev_loader)
        accuracy = sum(p == g for p, g in zip(predictions, goals)) / len(goals)
        print(f"训练第{epoch}轮，准确率：{accuracy:.4f}, 损失值: {total/ len(train_loader):.4f}")

    # 测试集测试
    predictions, goals = predict(model, test_loader)
    accuracy = sum(p == g for p, g in zip(predictions, goals)) / len(goals)
    f1 = macro_f1(predictions, goals, 12)
    print(f"测试集验证，准确率：{accuracy:.4f}, f1: {f1:.4f}")

    label, prop = predict_one("What is your name?", model, vocab=vocab, id2label=id2label)
    print(f"result is {label}")

if __name__ == "__main__":
    main()



