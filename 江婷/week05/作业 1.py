import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


"""
    python "作业 1.py" --sft --train-file sft_train/sft_train.jsonl --val-file sft_train/sft_val.jsonl --epochs 20 --ckpt sft_model.pt
    python "作业 1.py" --generate --sft --ckpt sft_model.pt --prompt "计算一组数字的平 均值和标准差。数字：23，45，67，89，12" --max-new 120
"""
# -------------------------- 模型模块 --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=x.device))
            attn = attn.masked_fill(mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, causal=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.max_seq_len:]
            logits = self(x_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)
        return x


# -------------------------- 数据集 1：原始字符预训练数据集 --------------------------
class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.data = text
        self.char2idx = char2idx
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        ids = [self.char2idx[c] for c in chunk]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


# -------------------------- 数据集 2：SFT Jsonl 数据集（核心！mask prompt loss） --------------------------
class SftJsonDataset(Dataset):
    def __init__(self, json_path, char2idx, seq_len):
        self.samples = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
        self.char2idx = char2idx
        self.seq_len = seq_len
        self.template_prompt = """### Instruction:
{instruction}
### Response:
"""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # 完整序列：prompt + answer
        prompt_str = self.template_prompt.format(instruction=item["instruction"])
        full_str = prompt_str + item["output"]

        # 转字符 id，过滤不在词表里的字符
        full_ids = [self.char2idx[c] for c in full_str if c in self.char2idx]
        prompt_ids = [self.char2idx[c] for c in prompt_str if c in self.char2idx]

        # 截断
        full_ids = full_ids[: self.seq_len + 1]
        prompt_len = len(prompt_ids)

        x = torch.tensor(full_ids[:-1], dtype=torch.long)
        y_raw = torch.tensor(full_ids[1:], dtype=torch.long)
        y = y_raw.clone()

        # ========== 重点偏移逻辑 ==========
        # x:      [P0, P1, P2, R0, R1, R2]
        # y_raw:  [P1, P2, R0, R1, R2, END]
        # prompt 在 x 里占 prompt_len 个 token
        # y 里前 prompt_len -1 个位置属于指令部分 → mask 掉，不计算 loss
        mask_end = prompt_len - 1
        if mask_end > 0:
            y[:mask_end] = -100

        # 补齐到 seq_len（方便batch）
        pad_len = self.seq_len - x.shape[0]
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=0)
            y = F.pad(y, (0, pad_len), value=-100)
        return x, y


# -------------------------- 构建字符词表 --------------------------
def build_vocab_from_files(file_list):
    chars = set()
    for fp in file_list:
        if fp.endswith(".jsonl"):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    chars.update(d["instruction"])
                    chars.update(d["output"])
        else:
            with open(fp, "r", encoding="utf-8") as f:
                chars.update(f.read())
    chars = sorted(list(chars))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    return char2idx, idx2char, len(chars)


# -------------------------- 主训练入口 --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", action="store_true", help="开启SFT模式，读取jsonl")
    parser.add_argument("--train-file", type=str)   # ✅ 去掉 required=True
    parser.add_argument("--val-file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default="model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 分支：生成模式（单独走，不加载训练文件、不重建词表） ==========
    if args.generate:
        # 加载 checkpoint，里面我们要存 vocab + 模型参数
        ckpt_data = torch.load(args.ckpt, map_location=device)
        char2idx = ckpt_data["char2idx"]
        idx2char = ckpt_data["idx2char"]
        vocab_size = ckpt_data["vocab_size"]

        model = TransformerLM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            max_seq_len=args.seq_len
        ).to(device)
        model.load_state_dict(ckpt_data["model"])
        model.eval()

        if args.sft:
            full_prompt = f"""### Instruction:
{args.prompt}
### Response:
"""
        else:
            full_prompt = args.prompt

        ids = [char2idx[c] for c in full_prompt if c in char2idx]
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(x, max_new_tokens=args.max_new)
        txt = "".join([idx2char[i.item()] for i in out[0]])
        print("\n===== Generate Result =====")
        print(txt)
        return

    # ========== 下面全部是训练模式，在这里校验 train-file ==========
    if args.train_file is None:
        parser.error("训练模式必须提供 --train-file")

    # 构建词表（训练集+验证集）
    vocab_files = [args.train_file]
    if args.val_file:
        vocab_files.append(args.val_file)
    char2idx, idx2char, vocab_size = build_vocab_from_files(vocab_files)
    print(f"Vocab size: {vocab_size}")

    # 加载数据集
    if args.sft:
        train_ds = SftJsonDataset(args.train_file, char2idx, args.seq_len)
        val_ds = SftJsonDataset(args.val_file, char2idx, args.seq_len) if args.val_file else None
    else:
        with open(args.train_file, "r", encoding="utf-8") as f:
            train_text = f.read()
        train_ds = CharDataset(train_text, char2idx, args.seq_len)
        val_ds = None
        if args.val_file:
            with open(args.val_file, "r", encoding="utf-8") as f:
                val_text = f.read()
            val_ds = CharDataset(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16) if val_ds else None

    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        max_seq_len=args.seq_len
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    best_ckpt_path = args.ckpt

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train_loss += loss.item()
        avg_train = total_train_loss / len(train_loader)

        avg_val = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
                    total_val_loss += loss.item()
            avg_val = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

            # Early stop：只保存验证 loss 最优的
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_data = {
                    "model": model.state_dict(),
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "vocab_size": vocab_size
                }
                torch.save(save_data, best_ckpt_path)
                print(f"  -> New best val loss, saved checkpoint to {best_ckpt_path}")
        else:
            print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train:.4f}")
            # 无验证集时，最后一轮保存
            if epoch == args.epochs - 1:
                save_data = {
                    "model": model.state_dict(),
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "vocab_size": vocab_size
                }
                torch.save(save_data, args.ckpt)
    print(f"Saved checkpoint to {args.ckpt}")


if __name__ == "__main__":
    main()
