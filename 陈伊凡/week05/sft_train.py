# -*- coding: utf-8 -*-
"""
在预训练字符级语言模型（language_model.py 训练的 best_model.pt）基础上做 SFT 微调。

数据:
    预训练: corpus_general.txt（language_model.py 已完成，产出 best_model.pt）
    SFT:    sft_train.jsonl / sft_val.jsonl  {"instruction": ..., "output": ...}

SFT 样本格式:
    输入 = ### 指令：{instruction}\n### 回答：{output}<eos>
    损失只在回答部分（含 <eos>）计算，指令部分 mask 为 -100。

用法:
    python sft_finetune.py                        # 用默认参数微调
    python sft_finetune.py --epochs 20 --lr 3e-4  # 自定义超参
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from language_model import TransformerLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROMPT_PREFIX = "### 指令："
PROMPT_SEP = "\n### 回答："
UNK, EOS = "<unk>", "<eos>"


# ─────────────────────────── 数据 ───────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode(text, char2idx):
    """字符 → id，生僻字映射为 <unk>。"""
    unk = char2idx[UNK]
    return [char2idx.get(c, unk) for c in text]


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows, char2idx, max_len=512):
        self.samples = []
        for r in rows:
            prompt = PROMPT_PREFIX + str(r.get("instruction", "")).strip() + PROMPT_SEP
            answer = str(r.get("output", "")).strip()
            if not answer:
                continue
            p_ids = encode(prompt, char2idx)
            a_ids = encode(answer, char2idx) + [char2idx[EOS]]
            if len(p_ids) >= max_len:          # prompt 被截断会错位，跳过
                continue
            ids = (p_ids + a_ids)[:max_len]
            labels = ([-100] * len(p_ids) + a_ids)[:len(ids)]
            if len(labels) < len(ids):
                labels += [-100] * (len(ids) - len(labels))
            self.samples.append((ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return torch.tensor(ids), torch.tensor(labels)


def collate(batch, pad_id=0):
    ids, labels = zip(*batch)
    maxlen = max(x.size(0) for x in ids)
    ids_t = torch.full((len(ids), maxlen), pad_id, dtype=torch.long)
    labels_t = torch.full((len(ids), maxlen), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(ids, labels)):
        ids_t[i, :x.size(0)] = x
        labels_t[i, :y.size(0)] = y
    return ids_t, labels_t


def sft_loss(logits, labels):
    """标准自回归错位：logits[i] 预测 labels[i+1]，-100 位置不计损失。"""
    return nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


# ─────────────────────────── 模型加载 ───────────────────────────

def load_pretrained(path, dropout=0.1):
    """加载预训练 checkpoint，并把词表扩展出 <unk> / <eos> 两行。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    char2idx = dict(ckpt["char2idx"])
    idx2char = dict(ckpt["idx2char"])
    for special in (UNK, EOS):
        if special not in char2idx:
            idx = len(char2idx)
            char2idx[special] = idx
            idx2char[idx] = special

    a = ckpt["args"]
    model = TransformerLM(
        vocab_size=len(char2idx),
        embed_dim=a["embed_dim"],
        num_layers=a["num_layers"],
        nhead=a["nhead"],
        ffn_dim=a["ffn_dim"],
        dropout=dropout,           # SFT 用小一点的 dropout，state_dict 里不存 dropout，可直接加载
    )
    old = ckpt["model_state"]
    new_sd = model.state_dict()
    n_old = old["embed.weight"].shape[0]
    # 复制预训练词向量/输出层，新增的 <unk>/<eos> 行随机初始化
    new_sd["embed.weight"][:n_old] = old["embed.weight"]
    nn.init.normal_(new_sd["embed.weight"][n_old:], mean=0.0, std=0.02)
    new_sd["fc.weight"][:n_old] = old["fc.weight"]
    nn.init.normal_(new_sd["fc.weight"][n_old:], mean=0.0, std=0.02)
    model.load_state_dict(new_sd)

    n_new = len(char2idx) - n_old
    print(f"已加载预训练模型（vocab {n_old} → {len(char2idx)}，扩展 {n_new} 个特殊字符）")
    return model, char2idx, idx2char, a


# ─────────────────────────── 训练 / 评估 ───────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tok = 0.0, 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        loss = sft_loss(model(ids), labels)
        n = (labels != -100).sum().item()
        total_loss += loss.item() * n
        total_tok += n
    model.train()
    return total_loss / max(total_tok, 1)


@torch.no_grad()
def generate(model, char2idx, idx2char, instruction, max_new=100, temperature=0.7, top_k=10):
    """自回归生成：每步把全部上下文喂给模型，遇到 <eos> 停止。"""
    model.eval()
    device = next(model.parameters()).device
    prompt = PROMPT_PREFIX + instruction.strip() + PROMPT_SEP
    ids = encode(prompt, char2idx)
    max_len = model.pos.num_embeddings
    for _ in range(max_new):
        if len(ids) >= max_len:
            break
        x = torch.tensor([ids[-max_len:]], device=device)
        logits = model(x)[0, -1] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
        if nxt == char2idx[EOS]:
            break
        ids.append(nxt)
    model.train()
    return "".join(idx2char[i] for i in ids[len(encode(prompt, char2idx)):])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",   default="best_model.pt", help="预训练模型 checkpoint")
    parser.add_argument("--train",    default="sft_train.jsonl")
    parser.add_argument("--val",      default="sft_val.jsonl")
    parser.add_argument("--epochs",   type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",       type=float, default=3e-4, help="SFT 学习率，比预训练小")
    parser.add_argument("--dropout",  type=float, default=0.1)
    parser.add_argument("--max_len",  type=int, default=512)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--save",     default="sft_model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, char2idx, idx2char, pre_args = load_pretrained(args.resume, args.dropout)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    print(f"SFT 训练样本 {len(train_rows)}，验证样本 {len(val_rows)}")

    train_loader = torch.utils.data.DataLoader(
        SFTDataset(train_rows, char2idx, args.max_len),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(
        SFTDataset(val_rows, char2idx, args.max_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda ep: max(0.1, 1.0 - 0.05 * ep))   # 每 epoch 衰减 5%，下限 10%

    best_val = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 46)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            loss = sft_loss(model(ids), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            nb += 1
        scheduler.step()
        val_loss = evaluate(model, val_loader, device)
        marker = "  *" if val_loss < best_val else ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": {**pre_args, "dropout": args.dropout, "sft": True,
                         "train": args.train, "val": args.val},
            }, args.save)
        print(f"{epoch:>6}  {running / max(nb,1):>10.4f}  {val_loss:>10.4f}  "
              f"{math.exp(min(val_loss, 20)):>10.2f}{marker}   ({time.time()-t0:.1f}s)")

    print(f"\nSFT 完成。最佳验证损失: {best_val:.4f}  已保存至 {args.save}")

    # ── 生成演示：训练集 / 验证集各抽 2 条 ──
    print("\n========== 生成演示（验证集） ==========")
    for r in random.sample(val_rows, 2):
        out = generate(model, char2idx, idx2char, r["instruction"])
        print(f"\n[指令] {r['instruction']}")
        print(f"[模型] {out}")
        print(f"[参考] {r['output']}")
    print("\n========== 生成演示（训练集） ==========")
    for r in random.sample(train_rows, 2):
        out = generate(model, char2idx, idx2char, r["instruction"])
        print(f"\n[指令] {r['instruction']}")
        print(f"[模型] {out}")
        print(f"[参考] {r['output']}")


if __name__ == "__main__":
    main()
