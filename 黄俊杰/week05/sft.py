# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from language_model import TransformerLM

"""
在 week04 预训练语言模型的基础上做 SFT：
    输入 = "指令：{instruction}\n回答：{output}<eos>"
    loss 只算回答部分，指令部分 mask 成 -100。
用法:
    python sft.py
    python sft.py --epochs 8 --lr 3e-4
    python sft.py --load sft_model.pt --prompt "计算 1+2 等于几？"
"""

EOS = "\u0003"          # 回答结束符，正常文本里不会出现
TEMPLATE = "指令：{}\n回答："
HERE = os.path.dirname(os.path.abspath(__file__))


# 数据

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_vocab(base_char2idx, rows):
    """在预训练词表后面补 SFT 数据里的新字和 <eos>。"""
    c2i = dict(base_char2idx)
    extra = {ch for r in rows for ch in
             TEMPLATE.format(r["instruction"]) + r["output"] + EOS}
    for ch in sorted(extra - set(c2i)):
        c2i[ch] = len(c2i)
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c


class SFTDataset(Dataset):
    """(instruction, output) -> ids 和 labels，prompt 部分不计 loss。"""

    def __init__(self, rows, char2idx, max_len):
        self.items = []
        for r in rows:
            prompt = TEMPLATE.format(r["instruction"].strip())
            answer = r["output"].strip() + EOS
            p = [char2idx.get(c, 0) for c in prompt]
            a = [char2idx.get(c, 0) for c in answer]
            ids = (p + a)[:max_len]
            labels = ([-100] * len(p) + a)[:max_len]
            if any(x != -100 for x in labels):
                self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    max_len = max(len(x) for x, _ in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
    return ids, labels


# 模型

def load_model(ckpt_path, char2idx, max_len, dropout=0.1):
    """加载 week04 的预训练权重，并把词表 / 位置编码扩到 SFT 需要的大小。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = TransformerLM(len(char2idx), max_len, a["dim"], a["nhead"],
                          a["hidden_dim"], a["num_layers"], dropout)

    old = ckpt["model_state"]
    new = model.state_dict()
    old_vocab = old["embed.weight"].shape[0]
    old_pos = old["pos.weight"].shape[0]
    for k, v in old.items():
        if k == "mask":
            continue
        if k in ("embed.weight", "fc.weight", "fc.bias"):
            new[k][:old_vocab] = v
        elif k == "pos.weight":
            new[k][:old_pos] = v
        else:
            new[k] = v

    # 新增的字 / 位置用小随机数初始化
    with torch.no_grad():
        new["embed.weight"][old_vocab:].normal_(0, 0.02)
        new["fc.weight"][old_vocab:].normal_(0, 0.02)
        new["fc.bias"][old_vocab:].zero_()
        new["pos.weight"][old_pos:].normal_(0, 0.02)
    model.load_state_dict(new)
    print(f"词表 {old_vocab} -> {len(char2idx)}，位置编码 {old_pos} -> {max_len}")
    return model, a


# 训练 / 生成

def sft_loss(logits, labels):
    return nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tok, correct = 0.0, 0, 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        loss = sft_loss(logits, labels)
        mask = labels[:, 1:] != -100
        n = mask.sum().item()
        total_loss += loss.item() * n
        total_tok += n
        correct += ((logits[:, :-1].argmax(-1) == labels[:, 1:]) & mask).sum().item()
    return total_loss / max(total_tok, 1), correct / max(total_tok, 1)


@torch.no_grad()
def generate(model, char2idx, idx2char, instruction, max_new=80,
             temperature=0.5, top_k=10):
    """按同样的模板拼 prompt，自回归生成到 <eos> 为止。"""
    model.eval()
    device = next(model.parameters()).device
    prompt = TEMPLATE.format(instruction.strip())
    ids = [char2idx.get(c, 0) for c in prompt]
    start = len(ids)
    max_len = model.pos.num_embeddings

    for _ in range(max_new):
        x = torch.tensor([ids[-max_len:]], device=device)
        logits = model(x)[0, -1] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
        if nxt == char2idx[EOS]:
            break
        ids.append(nxt)
    text = "".join(idx2char.get(i, "") for i in ids[start:]).strip()
    return text.lstrip("，。、；：,.;: ")


# 主函数

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default=os.path.join(HERE, "..", "week04", "best_model.pt"))
    parser.add_argument("--train",      default=os.path.join(HERE, "sft_train.jsonl"))
    parser.add_argument("--val",        default=os.path.join(HERE, "sft_val.jsonl"))
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--max_len",    type=int,   default=160)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save",       default=os.path.join(HERE, "sft_model.pt"))
    parser.add_argument("--load",       default="", help="加载 SFT 权重做生成")
    parser.add_argument("--prompt",     default=None, help="只生成，不训练")
    args = parser.parse_args()
    if args.load:
        args.ckpt = args.load

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    base = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    char2idx, idx2char = build_vocab(base["char2idx"], train_rows + val_rows)
    model, _ = load_model(args.ckpt, char2idx, args.max_len, args.dropout)
    model.to(device)

    if args.prompt:
        print(generate(model, char2idx, idx2char, args.prompt))
        return

    print(f"训练 {len(train_rows)} 条，验证 {len(val_rows)} 条，"
          f"参数量 {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(SFTDataset(train_rows, char2idx, args.max_len),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(SFTDataset(val_rows, char2idx, args.max_len),
                            batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)

    # SFT 之前：同一个 prompt，看它会不会「接着写」
    print("\n--- SFT 之前 ---")
    for r in val_rows[:2]:
        print(f"指令：{r['instruction']}")
        print(f"生成：{generate(model, char2idx, idx2char, r['instruction'])[:60]}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    log = []
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>8}")
    print("-" * 42)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tok = 0.0, 0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            loss = sft_loss(model(ids), labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = (labels[:, 1:] != -100).sum().item()
            total_loss += loss.item() * n
            total_tok += n

        va_loss, va_acc = evaluate(model, val_loader, device)
        log.append([total_loss / max(total_tok, 1), va_loss, va_acc])
        print(f"{epoch:>6}  {total_loss / max(total_tok,1):>10.4f}  "
              f"{va_loss:>10.4f}  {va_acc:>7.2%}")

    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "args": {
            "dim": base["args"]["dim"], "nhead": base["args"]["nhead"],
            "hidden_dim": base["args"]["hidden_dim"],
            "num_layers": base["args"]["num_layers"],
            "max_len": args.max_len, "template": TEMPLATE,
            "sft": True, "dropout": args.dropout,
        },
    }, args.save)
    print(f"\nSFT 完成，权重已保存到 {args.save}")

    # 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(log) + 1), [x[0] for x in log], label="train")
    plt.plot(range(1, len(log) + 1), [x[1] for x in log], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "sft-curve.png"), dpi=200)

    # SFT 之后
    print("\n--- SFT 之后 ---")
    for r in val_rows[:3]:
        print(f"指令：{r['instruction']}")
        print(f"参考：{r['output']}")
        print(f"生成：{generate(model, char2idx, idx2char, r['instruction'])}\n")


if __name__ == "__main__":
    main()
