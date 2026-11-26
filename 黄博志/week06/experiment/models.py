"""
模型加载与通用工具。

两个关键点：
1. 课程给的 RoBERTa 是 ModelScope 旧格式：
   - 权重 key 带 `bert.` 前缀；
   - LayerNorm 参数叫 `.gamma` / `.beta`，不是 HuggingFace 的 `.weight` / `.bias`。
   直接 from_pretrained 会静默丢参数，所以这里显式做转换，并断言没有 missing key。
2. 两个骨干都是 12 层 / 768 隐藏，参数量几乎一样（约 102M），
   这样“换模型”比较的是预训练数据/目标，而不是模型容量。
"""

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

import config


@lru_cache(maxsize=2)
def load_tokenizer(model_dir: str):
    """优先用目录里的 tokenizer 配置；没有就显式用 vocab.txt。"""
    try:
        return BertTokenizer.from_pretrained(model_dir)
    except Exception:
        return BertTokenizer(
            vocab_file=os.path.join(model_dir, "vocab.txt"),
            do_lower_case=True,
        )


def _clean_roberta_state_dict(model_dir: str):
    path = os.path.join(model_dir, "pytorch_model.bin")
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    clean = {}
    for k, v in sd.items():
        if not k.startswith("bert."):
            continue
        nk = k[len("bert."):]
        nk = nk.replace(".gamma", ".weight").replace(".beta", ".bias")
        clean[nk] = v
    return clean


def load_backbone(kind: str):
    """kind: 'bert' | 'roberta'。返回 eval 模式的 BertModel。"""
    if kind == "bert":
        model_dir = str(config.BERT_DIR)
        # 本地 bert-base-chinese 的 config 里 return_dict=False，
        # 这里强制打开，保证 last_hidden_state 可用。
        cfg = BertConfig.from_pretrained(model_dir)
        cfg.return_dict = True
        model = BertModel.from_pretrained(model_dir, config=cfg)
    elif kind == "roberta":
        model_dir = str(config.ROBERTA_DIR)
        cfg = BertConfig.from_pretrained(model_dir)
        cfg.return_dict = True
        model = BertModel(cfg)
        clean = _clean_roberta_state_dict(model_dir)
        missing, unexpected = model.load_state_dict(clean, strict=False)
        if missing:
            raise RuntimeError(
                f"RoBERTa 权重转换后仍有 missing keys（{len(missing)} 个）：{missing[:5]}"
            )
        if unexpected:
            print(f"[warn] RoBERTa unexpected keys: {unexpected[:5]}")
    else:
        raise ValueError(f"未知骨干：{kind}")

    model.eval()
    return model


class BertClassifier(nn.Module):
    """骨干 + dropout + 线性分类头，和课程里的结构保持一致。"""

    def __init__(self, backbone: BertModel, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.backbone(**kwargs)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        cls = h[:, 0]
        return self.classifier(self.dropout(cls))


def build_classifier(kind: str, num_labels: int):
    return BertClassifier(load_backbone(kind), num_labels)


def freeze_backbone(model: BertClassifier):
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: BertClassifier):
    for p in model.backbone.parameters():
        p.requires_grad = True


@torch.no_grad()
def encode_texts(
    model: BertModel,
    tokenizer,
    texts,
    max_len: int = 40,
    batch_size: int = 64,
    pool: str = "cls",
    normalize: bool = True,
    device: str = "cpu",
):
    """把文本编码成句向量，默认 [CLS] + L2 归一化。"""
    model.eval()
    model.to(device)
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if pool == "cls":
            v = h[:, 0]
        elif pool == "mean":
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        elif pool == "max":
            mask = enc["attention_mask"].unsqueeze(-1).bool()
            v = h.masked_fill(~mask, -1e9).max(dim=1).values
        else:
            raise ValueError(f"未知 pool：{pool}")
        if normalize:
            v = nn.functional.normalize(v, p=2, dim=1)
        feats.append(v.cpu())
    return torch.cat(feats, dim=0).numpy()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / (1024 ** 2), 1)
