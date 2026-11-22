# -*- coding: utf-8 -*-


import math
import argparse
import glob
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))

# 默认 --corpus="*.txt"：当前目录找不到时，再去脚本目录和课件目录找
DEFAULT_CORPUS_CANDIDATES = [
    os.path.join(HERE, "*.txt"),
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "..",
                                 "week4-语言模型",
                                 "循环神经网络语言模型", "corpus_general.txt")),
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "..",
                                 "第四周 语言模型", "week4-语言模型",
                                 "循环神经网络语言模型", "corpus_general.txt")),
]


def resolve_corpus(pattern):
    """
    把 --corpus 解析成一个确实能匹配到文件的 glob；返回 None 表示没找到。
    只有用户没显式指定（还是默认的 *.txt）时，才去候选位置找，
    避免覆盖用户明确写的路径。
    """
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, "*.txt")
    if glob.glob(pattern):
        return pattern
    if pattern == "*.txt":
        for cand in DEFAULT_CORPUS_CANDIDATES:
            if glob.glob(cand):
                return cand
    return None


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1        —— 每个字符位置都当一个样本（重叠严重，语料大时一轮极慢）
        stride=seq_len  —— 不重叠切块（标准做法）
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

def causal_mask(T, device):
    """
    上三角（不含对角线）= 所有「未来」位置。

    torch.triu(ones, diagonal=1) 得到的 True 表示「这个位置不该被看到」；
    而 SDPA 的 bool attn_mask 里 True 表示「允许参与」，语义正好相反，
    所以返回 ~future（下三角 + 对角线为 True）。
    """
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    return ~future


class CausalSelfAttention(nn.Module):
    """多头因果自注意力：用 scaled_dot_product_attention 实现。"""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0, f"d_model({d_model}) 必须能被 n_head({n_head}) 整除"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, mask):
        B, T, C = x.shape

        # (B, T, 3C) -> 3 个 (B, n_head, T, d_head)
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # mask: (T, T) bool，True = 允许看；自动广播到 (B, n_head, T, T)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mult * d_model)
        self.fc2 = nn.Linear(mult * d_model, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class Block(nn.Module):
    """Pre-LN 的 Transformer block：先归一化再进子层，残差直接加。"""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, 4, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 n_head, dropout=0.1, max_seq_len=1024):
        super().__init__()
        assert hidden_dim % n_head == 0, f"hidden_dim({hidden_dim}) 必须能被 n_head({n_head}) 整除"
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.in_proj = nn.Linear(embed_dim, hidden_dim)      # 把 embed 维对齐到 d_model
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)  # 可学习位置编码
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(hidden_dim, n_head, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2 的残差分支缩放：层数越多，残差分支初始化越小，训练越稳
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=0.02 / math.sqrt(2 * num_layers))
            nn.init.normal_(blk.mlp.fc2.weight, std=0.02 / math.sqrt(2 * num_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        if T > self.max_seq_len:
            raise ValueError(f"序列长度 {T} 超过 max_seq_len={self.max_seq_len}")

        pos = torch.arange(T, device=x.device)
        h = self.drop(self.in_proj(self.embed(x)) + self.pos_emb(pos))

        mask = causal_mask(T, x.device)      # 每层共用同一张因果掩码
        for blk in self.blocks:
            h = blk(h, mask)

        return self.fc(self.drop(self.ln_f(h)))   # (B, T, V)


# ─────────────────────────── 因果性自检 ───────────────────────────

@torch.no_grad()
def check_causality(model, seq_len=16, device="cpu"):
    """
    把后半段输入改掉，看前半段输出是否变化。
    正确的因果掩码下，位置 t 的输出只依赖 <= t 的输入，所以必须完全一致。
    """
    model.eval()
    V = model.fc.out_features
    x = torch.randint(1, V, (1, seq_len), device=device)
    y1 = model(x)

    x2 = x.clone()
    x2[:, seq_len // 2:] = torch.randint(1, V, (1, seq_len - seq_len // 2), device=device)
    y2 = model(x2)

    k = seq_len // 2
    diff = (y1[:, :k] - y2[:, :k]).abs().max().item()
    print(f"  因果性检查：改动后半段后，前 {k} 个位置输出的最大差异 = {diff:.2e}")
    if diff > 1e-5:
        raise SystemExit("因果掩码失效：前面的位置被后面的输入影响了！")
    print("  ✅ 通过：位置 t 只能看到 <= t 的输入")


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True,
              grad_clip=1.0, log_every=0, tag="train"):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        # 训练时定期打印进度，避免"跑了很久没有任何输出"
        if log_every and step % log_every == 0:
            avg = total_loss / total_tokens
            elapsed = time.time() - t0
            eta = elapsed / step * (len(loader) - step)
            print(f"    {tag} {step:>6}/{len(loader)}  loss={avg:.4f}  "
                  f"{elapsed:>6.0f}s  ETA {eta/60:>6.1f}min", flush=True)

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64,
                        help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256, help="Transformer 的 d_model")
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--n_head",     type=int,   default=8, help="必须整除 hidden_dim")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--grad_clip",  type=float, default=1.0)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt",
                        help="语料 glob 或目录；默认先找当前目录，再找脚本目录和课件目录")
    parser.add_argument("--save",       default="best_model_transformer.pt")
    parser.add_argument("--check_causal", action="store_true", help="只做因果性自检")
    parser.add_argument("--log_every", type=int, default=50,
                        help="每 N 个 batch 打印一次训练进度，0=只打印每个 epoch 的结果")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER")

    # 数据准备
    corpus = resolve_corpus(args.corpus)
    if corpus is None:
        tried = "、".join([args.corpus] + DEFAULT_CORPUS_CANDIDATES)
        raise FileNotFoundError(
            f"找不到语料：'{args.corpus}'。\n"
            f"  已尝试：{tried}\n"
            f"  请显式指定：--corpus /path/to/corpus_general.txt"
        )
    if corpus != args.corpus:
        print(f"未指定 --corpus，自动使用：{corpus}")
    text = load_corpus(corpus)
    if not text:
        raise FileNotFoundError(f"语料文件为空：{corpus}")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"训练 batch 数: {len(train_loader):,}   验证 batch 数: {len(val_loader):,}")
    if len(train_loader) > 3000:
        print(f"[提示] 训练 batch 很多（stride={args.stride}），一轮会跑很久；"
              f"想快就把 --stride 设成 {args.seq_len}（不重叠切块）")

    # 模型
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_head=args.n_head,
        dropout=args.dropout,
        max_seq_len=max(args.seq_len, 64),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    if args.check_causal:
        check_causality(model, seq_len=min(16, model.max_seq_len), device=device)
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    saved = False

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer,
                                    device, train=True, grad_clip=args.grad_clip,
                                    log_every=args.log_every, tag="train")
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer,
                                        device, train=False, log_every=0)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)
            saved = True

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    tail = f"  已保存至 {args.save}" if saved else "  （没有完成任何 epoch，未保存模型）"
    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}{tail}")


if __name__ == "__main__":
    main()