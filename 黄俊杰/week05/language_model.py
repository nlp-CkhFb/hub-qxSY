# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import glob
import random
import argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

"""
字符级 Transformer 语言模型：输入 seq_len 个字，预测下一个字。
用法:
    python language_model.py
    python language_model.py --epochs 5 --batch_size 64
"""


# 数据

def load_corpus(pattern="corpus_general.txt"):
    text = ""
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text += f.read()
    return text


def build_vocab(text, max_vocab=None):
    """按出现频率取前 max_vocab 个字做字表，None 表示全部字符。"""
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """x = 连续 seq_len 个字，y = 往后错一位的字；stride 是切块间隔。"""

    def __init__(self, text, char2idx, seq_len, stride=None):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# 模型

class Block(nn.Module):
    """自注意力 + 前馈网络，均带残差，LayerNorm 放在子层前（Pre-LN）。"""

    def __init__(self, dim, nhead, hidden_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n, attn_mask=mask, need_weights=False)   # q = k = v
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, seq_len, dim=128, nhead=4, hidden_dim=256,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(seq_len, dim)      # 位置编码
        self.blocks = nn.ModuleList([Block(dim, nhead, hidden_dim, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, vocab_size)

        # 因果掩码：上三角置 -inf，每个位置只能看左边
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, t = x.shape
        h = self.embed(x) + self.pos(torch.arange(t, device=x.device))   # (B, T, dim)
        for block in self.blocks:
            h = block(h, self.mask[:t, :t])
        return self.fc(self.norm(h))                                    # (B, T, vocab)


# 训练 / 评估

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """跑一轮，返回 (平均 loss, PPL)。"""
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪
            optimizer.step()
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)    # PPL = e^loss


# 生成

@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, n=40, temperature=1.0, device="cpu"):
    """给开头几个字，按 temperature 采样续写 n 个字。"""
    model.eval()
    seq_len = model.pos.num_embeddings
    ids = [char2idx.get(c, 0) for c in prompt]
    for _ in range(n):
        x = torch.tensor([ids[-seq_len:]], device=device)
        logits = model(x)[0, -1] / temperature      # 取最后一个位置的输出
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        ids.append(nxt)
    return "".join(idx2char[i] for i in ids)


# 主函数

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",     default="corpus_general.txt")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--dim",        type=int,   default=128)
    parser.add_argument("--nhead",      type=int,   default=4)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save",       default="best_model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 数据
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"未找到语料: {args.corpus}")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text, val_text = "\n".join(lines[:split]), "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"训练样本: {len(train_ds):,}  验证样本: {len(val_ds):,}")

    # 模型
    model = TransformerLM(vocab_size, args.seq_len, args.dim, args.nhead,
                          args.hidden_dim, args.num_layers, args.dropout).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    log = []
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        log.append([tr_ppl, va_ppl])

        # 按验证集 PPL 保存最优权重
        marker = ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            marker = "  *"
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}，权重已保存到 {args.save}")

    # 训练曲线
    epochs = range(1, len(log) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [l[0] for l in log], label="train")
    plt.plot(epochs, [l[1] for l in log], label="val")
    plt.xlabel("epoch")
    plt.ylabel("PPL")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [math.log(l[0]) for l in log], label="train")
    plt.plot(epochs, [math.log(l[1]) for l in log], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("train-curve.png", dpi=200)
    plt.show()

    # 用最优权重续写
    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    for prompt in ["人工智能", "今天天气", "我想问一下"]:
        print(f"{prompt} -> {generate(model, char2idx, idx2char, prompt, n=30, temperature=0.8, device=device)}")


if __name__ == "__main__":
    main()
