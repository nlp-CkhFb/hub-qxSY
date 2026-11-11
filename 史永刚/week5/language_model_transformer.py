# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── 模型 ───────────────────────────

def causal_mask(T, device):
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    return ~future


class CausalSelfAttention(nn.Module):
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

    def forward(self, x, key_padding_mask=None):
        B, T = x.shape
        if T > self.max_seq_len:
            raise ValueError(f"序列长度 {T} 超过 max_seq_len={self.max_seq_len}")

        pos = torch.arange(T, device=x.device)
        h = self.drop(self.in_proj(self.embed(x)) + self.pos_emb(pos))

        mask = causal_mask(T, x.device).view(1, 1, T, T)
        if key_padding_mask is not None:
            keep = (~key_padding_mask).view(B, 1, 1, T)
            mask = mask & keep                      # (B, 1, T, T)

        for blk in self.blocks:
            h = blk(h, mask)

        return self.fc(self.drop(self.ln_f(h)))   # (B, T, V)
