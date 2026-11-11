# -*- coding: utf-8 -*-
import os
import json
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from language_model_transformer import TransformerLM

HERE = os.path.dirname(os.path.abspath(__file__))

EOS = "\u0003"          # 回答结束符（不会出现在正常文本里）
UNK = "\u0002"          # 词表外的字
TEMPLATE = "指令：{instruction}\n回答："


# ─────────────────────────── 词表 ───────────────────────────

def build_sft_vocab(base_c2i, rows):
    """
    在预训练词表后面追加：SFT 数据里的新字 + <UNK> + <EOS>。
    返回 (新 char2idx, idx2char, 新增字数)。
    """
    base = set(base_c2i)
    new_chars = sorted({c for r in rows for c in r["instruction"] + r["output"]}
                       - base - {EOS, UNK})
    c2i = dict(base_c2i)
    for ch in new_chars:
        c2i[ch] = len(c2i)
    c2i[UNK] = len(c2i)
    c2i[EOS] = len(c2i)
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c, len(new_chars)


def resize_model(model, old_vocab_size, old_embed, old_fc, old_pos, new_pos_len):
    """把 embedding / 输出层 / 位置编码扩到新尺寸，旧权重原样保留。"""
    with torch.no_grad():
        model.embed.weight[:old_vocab_size] = old_embed
        model.fc.weight[:old_vocab_size] = old_fc
        n = min(old_pos.size(0), new_pos_len)
        model.pos_emb.weight[:n] = old_pos[:n]
    return model


# ─────────────────────────── 数据 ───────────────────────────

class SFTDataset(Dataset):
    """把 (instruction, output) 拼成一条序列，并给指令部分打 -100。"""

    def __init__(self, rows, c2i, max_len):
        self.unk = c2i[UNK]
        self.eos = c2i[EOS]
        self.items = []
        dropped = 0
        for r in rows:
            prompt = TEMPLATE.format(instruction=r["instruction"])
            p = [c2i.get(c, self.unk) for c in prompt]
            a = [c2i.get(c, self.unk) for c in r["output"]] + [self.eos]

            ids = (p + a)[:max_len]
            labels = ([-100] * len(p) + a)[:max_len]

            if all(x == -100 for x in labels):   # 答案被截没了，丢掉
                dropped += 1
                continue
            self.items.append((ids, labels))
        self.dropped = dropped

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    L = max(len(x) for x, _ in batch)
    X = torch.zeros(len(batch), L, dtype=torch.long)
    Y = torch.full((len(batch), L), -100, dtype=torch.long)
    lengths = []
    for i, (x, y) in enumerate(batch):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
        lengths.append(len(x))

    lens = torch.tensor(lengths)
    pad_mask = torch.arange(L).unsqueeze(0) >= lens.unsqueeze(1)   # True = 补齐位
    return X, Y, pad_mask


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, optimizer, device, train=True, grad_clip=1.0, log_every=20):
    model.train(train)
    total, tokens = 0.0, 0
    t0 = time.time()

    for step, (X, Y, pad) in enumerate(loader, 1):
        X, Y, pad = X.to(device), Y.to(device), pad.to(device)
        logits = model(X, key_padding_mask=pad)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               Y[:, 1:].reshape(-1), ignore_index=-100)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if log_every and step % log_every == 0:
                el = time.time() - t0
                print(f"    step {step:>4}/{len(loader)}  loss={total/max(tokens,1):.4f}  "
                      f"{el:.0f}s  ETA {el/step*(len(loader)-step)/60:.1f}min", flush=True)

        n = (Y[:, 1:] != -100).sum().item()
        total += loss.item() * n
        tokens += n

    return total / max(tokens, 1)


@torch.no_grad()
def generate(model, c2i, i2c, instruction, max_new=80, eos_id=None, greedy=True):
    model.eval()
    prompt = TEMPLATE.format(instruction=instruction)
    ids = [c2i.get(c, c2i[UNK]) for c in prompt]
    start = len(ids)

    for _ in range(max_new):
        x = torch.tensor([ids[-model.max_seq_len:]])
        logits = model(x)[0, -1]
        nxt = int(logits.argmax()) if greedy else int(torch.multinomial(logits.softmax(-1), 1))
        if eos_id is not None and nxt == eos_id:
            break
        ids.append(nxt)
        if i2c.get(nxt) == "\n" and len(ids) > start:   # 没 EOS 的旧模型用换行停
            break
    return "".join(i2c.get(i, "") for i in ids[start:]).strip()


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "best_model_transformer.pt"),
                    help="预训练模型路径（默认本目录的 best_model_transformer.pt）")
    ap.add_argument("--train", default=os.path.join(HERE, "sft_train", "sft_train.jsonl"))
    ap.add_argument("--val", default=os.path.join(HERE, "sft_train", "sft_val.jsonl"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=os.path.join(HERE, "sft_model.pt"))
    ap.add_argument("--infer_only", default=None, help="跳过训练，直接用这个指令生成")
    ap.add_argument("--load_sft", default=None, help="加载训练好的 SFT 模型，只做推理")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 0. 只推理：加载已经 SFT 好的模型 ──
    if args.load_sft:
        ck2 = torch.load(args.load_sft, map_location="cpu", weights_only=False)
        c2i, i2c = ck2["char2idx"], ck2["idx2char"]
        a2 = ck2["args"]
        model = TransformerLM(len(c2i), a2["embed_dim"], a2["hidden_dim"],
                              a2["num_layers"], a2["n_head"], a2["dropout"],
                              max_seq_len=a2.get("max_len", 128))
        model.load_state_dict(ck2["model_state"])
        model.to(device).eval()
        print(f"已加载 SFT 模型：{args.load_sft}（词表 {len(c2i)}）\n")
        prompts = [args.infer_only] if args.infer_only else [
            "将以下句子转换为被动语态。 句子： Tom 发现了一份工作。",
            "给定两个数，判断它们是否互质。 数字1：12，数字2：25",
            "给定一组数据，判断其中是否存在异常值。 数据组：1,2,3,4,5,6,1000",
        ]
        for p in prompts:
            print(f"  指令：{p}")
            print(f"  生成：{generate(model, c2i, i2c, p, eos_id=c2i[EOS])}\n")
        return

    # ── 1. 加载预训练模型 ──
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"找不到预训练模型：{args.ckpt}\n"
                         f"  请把 best_model_transformer.pt 放在本目录下")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    old_c2i = ck["char2idx"]
    old_v = len(old_c2i)
    sd = ck["model_state"]
    old_embed = sd["embed.weight"].clone()
    old_fc = sd["fc.weight"].clone()
    old_pos = sd["pos_emb.weight"].clone()

    train_rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    val_rows = [json.loads(l) for l in open(args.val, encoding="utf-8") if l.strip()]

    c2i, i2c, n_new = build_sft_vocab(old_c2i, train_rows)
    V = len(c2i)
    print(f"预训练词表 {old_v} → SFT 词表 {V}（新增 {n_new} 个字符 + <UNK> + <EOS>）")
    print(f"预训练位置编码 {old_pos.size(0)} → {args.max_len}")

    model = TransformerLM(V, a["embed_dim"], a["hidden_dim"], a["num_layers"],
                          a["n_head"], a["dropout"], max_seq_len=args.max_len)
    for k in ("embed.weight", "fc.weight", "pos_emb.weight"):
        sd.pop(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    resize_model(model, old_v, old_embed, old_fc, old_pos, args.max_len)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = SFTDataset(train_rows, c2i, args.max_len)
    val_ds = SFTDataset(val_rows, c2i, args.max_len)
    print(f"SFT 样本：训练 {len(train_ds)} / 验证 {len(val_ds)}"
          + (f"（丢弃 {train_ds.dropped} 条超长样本）" if train_ds.dropped else ""))

    # ── 2. 训练前先看一眼（SFT 之前模型只会"接着写"）──
    if args.infer_only is None:
        print("\n--- SFT 之前 ---")
        for r in val_rows[:3]:
            out = generate(model, c2i, i2c, r["instruction"])
            print(f"  指令：{r['instruction'][:30]}\n  回答：{r['output'][:40]}\n  生成：{out[:60]}")

    # ── 3. 训练 ──
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.infer_only is None:
        print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}")
        print("-" * 32)
        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, opt, device, train=True)
            with torch.no_grad():
                va = run_epoch(model, val_loader, opt, device, train=False)
            print(f"{epoch:>6}  {tr:>10.4f}  {va:>10.4f}")

        torch.save({
            "model_state": model.state_dict(),
            "char2idx": c2i,
            "idx2char": i2c,
            "args": {**a, **vars(args)},
            "sft": True,
        }, args.save)
        print(f"\nSFT 模型已保存：{args.save}")

    # ── 4. 推理 ──
    print("\n--- SFT 之后 ---")
    prompts = [args.infer_only] if args.infer_only else [r["instruction"] for r in val_rows[:5]]
    for p in prompts:
        print(f"  指令：{p[:40]}")
        print(f"  生成：{generate(model, c2i, i2c, p, eos_id=c2i[EOS])[:80]}")


if __name__ == "__main__":
    main()
