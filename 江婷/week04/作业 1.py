"""
字符级 Transformer Decoder 语言模型
接口、评估、数据流程和原版 RNN/LSTM 脚本对齐，含 PPL、文本生成
用法:
    python lm_transformer.py --epochs 20 --n_layer 4 --n_head 4
"""
import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────——
def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
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


# ─────────────────────────── Transformer 组件 ───────────────────────────
class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, embed_dim, n_head, dropout):
        super().__init__()
        assert embed_dim % n_head == 0, "embed_dim 必须可以被 n_head 整除"
        self.n_head = n_head
        self.head_dim = embed_dim // n_head

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B,T,3C)
        q, k, v = torch.split(qkv, C, dim=-1)

        # 切多头
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadMaskedAttention(embed_dim, n_head, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_head, n_layer, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, n_head, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # 预缓存上三角掩码，训练时固定
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)[None, :]
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(pos)
        x = self.drop(token_emb + pos_emb)

        for block in self.blocks:
            x = block(x, self.mask[:T, :T])

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """推理生成，作业提交用"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


# ─────────────────────────── 训练评估 ───────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主入口 ───────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=1)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--n_head",     type=int,   default=4, help="多头注意力头数")
    parser.add_argument("--n_layer",    type=int,   default=4, help="Decoder 层数")
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_transformer.pt")
    parser.add_argument("--gen",        action="store_true", help="加载模型生成文本")
    parser.add_argument("--prompt",     default="")
    parser.add_argument("--max_gen",    type=int,   default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 如果只是生成
    if args.gen:
        ckpt = torch.load(args.save, map_location=device)
        char2idx = ckpt["char2idx"]
        idx2char = ckpt["idx2char"]
        cfg = ckpt["args"]
        model = TransformerLM(
            vocab_size=len(char2idx),
            embed_dim=cfg["embed_dim"],
            n_head=cfg["n_head"],
            n_layer=cfg["n_layer"],
            seq_len=cfg["seq_len"],
            dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        if not args.prompt:
            args.prompt = "Once upon a time"
        idx = torch.tensor([[char2idx[c] for c in args.prompt if c in char2idx]], dtype=torch.long, device=device)
        out_idx = model.generate(idx, args.max_gen)
        out_text = "".join(idx2char[int(i)] for i in out_idx[0])
        print("\n===== 生成结果 =====")
        print(out_text)
        return

    # -------- 训练流程 --------
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        n_head=args.n_head,
        n_layer=args.n_layer,
        seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
