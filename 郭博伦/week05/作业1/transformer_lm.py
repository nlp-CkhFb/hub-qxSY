"""字符级 Transformer 语言模型 —— 两阶段：预训练（续写）+ 指令微调（SFT）。"""

import argparse
import glob
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))

SEP = "\n回答："


# ───────────────────────── 预训练数据（续写 LM） ─────────────────────────

def load_corpus(pattern):
    text = ""
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8", errors="ignore") as f:
            text += f.read()
    if not text:
        raise FileNotFoundError(f"没读到语料: {pattern}")
    return text


class CharDataset(Dataset):
    """x = 连续 seq_len 字符，y = 往后错一位；按 stride 切块。"""

    def __init__(self, text, char2idx, seq_len, stride):
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = max(1, stride)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, i):
        s = i * self.stride
        return self.data[s:s + self.seq_len], self.data[s + 1:s + self.seq_len + 1]


# ───────────────────────── SFT 数据（指令微调） ─────────────────────────

def load_sft(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "instruction" in obj and "output" in obj:
                pairs.append((obj["instruction"], obj["output"]))
    return pairs


class SFTDataset(Dataset):
    """
    每条拼成 instruction + SEP + output；目标是错一位的下一个字符。
    loss_mask 只在 output 段（含 SEP 那一位）为 1，其余为 0。
    超长统一截断到 seq_len，保证 batch 内所有 tensor 等长。
    """

    def __init__(self, pairs, char2idx, seq_len):
        self.seq_len = seq_len
        self.items = []
        for ins, out in pairs:
            # 逐个字符同步构建 ids 与 mask，字表外的字符一起跳过，
            # 保证 ids 和 mask 长度严格一致（否则后面截断/补齐会错位）
            ids, mask = [], []
            for c in ins:
                if c in char2idx:
                    ids.append(char2idx[c])
                    mask.append(0)
            for c in SEP + out:
                if c in char2idx:
                    ids.append(char2idx[c])
                    mask.append(1)
            # 统一截断到 seq_len，保证 batch 内所有样本等长
            ids = ids[:self.seq_len]
            mask = mask[:self.seq_len]
            if len(ids) < 2:
                continue
            self.items.append((ids, mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, mask = self.items[i]
        n = len(ids)
        # x 取 [0, n)，y 取 [1, n+1)（最后一个 y 用 0 补齐），都补齐到 seq_len
        x = ids + [0] * (self.seq_len - n)
        y = (ids[1:] + [0]) + [0] * (self.seq_len - n)
        m = mask + [0] * (self.seq_len - n)
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(m, dtype=torch.long))


# ───────────────────────── 模型 ─────────────────────────

def build_vocab(chars_iter, max_vocab):
    from collections import Counter
    counter = Counter(chars_iter)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(counter)
    char2idx = {c: i for i, c in enumerate(chars)}
    return char2idx, {i: c for c, i in char2idx.items()}


class TransformerLM(nn.Module):
    """可学习位置编码 + N 层 TransformerEncoderLayer + 输出头（与输入 embedding 共享权重）。"""

    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=128, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight

        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx):
        x = self.tok(idx)
        x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(idx.size(1), device=idx.device)
        x = self.encoder(x, mask=mask)
        return self.head(x)


def masked_loss(logits, y, mask):
    """只在 mask=1 的位置算交叉熵。"""
    logits = logits.reshape(-1, logits.size(-1))
    y = y.reshape(-1)
    mask = mask.reshape(-1)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    loss = F.cross_entropy(logits, y, reduction="none")
    return (loss * mask).sum() / mask.sum()


# ───────────────────────── 训练流程 ─────────────────────────

def pretrain(args, device):
    """阶段一：用 corpus_sample.txt 预训练续写 LM，存 tf_best.pt。"""
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    text = load_corpus(args.corpus)
    if args.max_chars and len(text) > args.max_chars:
        text = text[:args.max_chars]
    char2idx, idx2char = build_vocab(text, args.max_vocab)
    V = len(char2idx)
    print(f"[阶段一 预训练] device: {device}  语料 {len(text):,} 字  字表 {V}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    tr = CharDataset("\n".join(lines[:split]), char2idx, args.seq_len, args.stride)
    va = CharDataset("\n".join(lines[split:]), char2idx, args.seq_len, args.stride)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TransformerLM(V, args.d_model, args.heads, args.layers, args.seq_len, args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型: TransformerEncoderLayer × {args.layers}（d={args.d_model} h={args.heads} "
          f"FFN={4 * args.d_model} Pre-LN）  参数量 {n_par / 1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPPL':>10} {'tok/s':>9}")
    print("-" * 42)
    for ep in range(1, args.pt_epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * y.numel()
            seen += y.numel()
        dt = time.time() - st
        _, v_ppl = evaluate_pt(model, val_loader, device)
        mark = ""
        if v_ppl < best:
            best = v_ppl
            mark = " *"
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": vars(args)}, args.save)
        print(f"{ep:>6} {tot / max(seen, 1):>10.4f} {v_ppl:>10.2f} {seen / dt:>9,.0f}{mark}")

    print(f"\n预训练完成，最佳验证 PPL: {best:.2f}  已保存 {args.save}")
    return model, char2idx, idx2char


@torch.no_grad()
def evaluate_pt(model, loader, device):
    model.eval()
    total, tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1), reduction="sum"))
        tokens += y.numel()
    loss = total / max(tokens, 1)
    return loss, math.exp(loss)


def sft(args, device, model, char2idx, idx2char):
    """阶段二：用 sft_train 指令微调，更新 tf_best.pt。"""
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_pairs = load_sft(args.train_file)
    val_pairs = load_sft(args.val_file)
    print(f"[阶段二 SFT] 训练样本 {len(train_pairs)}  验证样本 {len(val_pairs)}")

    tr = SFTDataset(train_pairs, char2idx, args.seq_len)
    va = SFTDataset(val_pairs, char2idx, args.seq_len)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.sft_lr)
    best = float("inf")
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} {'ValPPL':>10} {'tok/s':>9}")
    print("-" * 50)
    for ep in range(1, args.sft_epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = masked_loss(model(x), y, m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * int(m.sum())
            seen += int(m.sum())
        dt = time.time() - st
        v_loss, v_ppl = evaluate_sft(model, val_loader, device)
        mark = ""
        if v_ppl < best:
            best = v_ppl
            mark = " *"
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": vars(args)}, args.save)
        print(f"{ep:>6} {tot / max(seen, 1):>10.4f} {v_loss:>10.4f} {v_ppl:>10.2f} {seen / dt:>9,.0f}{mark}")

    print(f"\nSFT 完成，最佳验证 PPL: {best:.2f}  已保存 {args.save}")


@torch.no_grad()
def evaluate_sft(model, loader, device):
    model.eval()
    total, tokens = 0.0, 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        total += float((loss * m.reshape(-1)).sum())
        tokens += int(m.sum())
    loss = total / max(tokens, 1)
    return loss, math.exp(loss)


@torch.no_grad()
def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    model = TransformerLM(len(char2idx), c["d_model"], c["heads"], c["layers"], c["seq_len"], 0.0).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    prompt = args.generate + SEP
    ids = [char2idx[ch] for ch in prompt if ch in char2idx]
    out = list(ids)
    for _ in range(args.n_chars):
        logits = model(torch.tensor([out[-args.seq_len:]], device=device))[0, -1]
        if args.temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            nxt = int(torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1))
        out.append(nxt)
    print(f"输入: {args.generate}")
    print(f"续写: {''.join(idx2char[i] for i in out[len(ids):])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus_sample.txt"))
    ap.add_argument("--train_file", default=os.path.join(HERE, "sft_train", "sft_train.jsonl"))
    ap.add_argument("--val_file", default=os.path.join(HERE, "sft_train", "sft_val.jsonl"))
    ap.add_argument("--max_chars", type=int, default=0)
    ap.add_argument("--max_vocab", type=int, default=4000)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sft_lr", type=float, default=5e-4)
    ap.add_argument("--pt_epochs", type=int, default=3)
    ap.add_argument("--sft_epochs", type=int, default=5)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default="tf_best.pt")
    ap.add_argument("--generate", default=None, help="给一段 instruction，续写回答")
    ap.add_argument("--ckpt", default="tf_best.pt")
    ap.add_argument("--n_chars", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    if args.generate:
        generate(args)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, char2idx, idx2char = pretrain(args, device)
        sft(args, device, model, char2idx, idx2char)


if __name__ == "__main__":
    main()
