"""
模型工厂与推理工具。

与课程原版 common/models.py 的差别只有两点：
  1. 骨干由 src/backbones.py 解析，支持 BERT / RoBERTa 两个本地目录；
  2. predict_char_tags 除了字符级标签，还返回 token 级路径与句级置信度，
     这样「token 级非法率」和「字符级非法率」能同时报（课程里这两个口径结论相反）。
"""

import torch
import torch.nn as nn

from backbones import TransformerEncoder
from crf import CRFClassifier
from event_data import (
    IGNORE_INDEX,
    build_bio_constraint_mask,
    build_start_constraint_mask,
)


def build_model(backbone_path: str, label_scheme: dict, decoder: str = "crf",
                num_layers: int = 0, dropout: float = 0.1,
                hard_constraint: bool = False) -> CRFClassifier:
    """构造 encoder + (Linear|CRF)。hard_constraint 只影响解码，不影响训练 loss。"""
    encoder = TransformerEncoder(backbone_path, num_layers=num_layers, dropout=dropout)
    use_crf = decoder.lower() == "crf"
    constraint = build_bio_constraint_mask(label_scheme) if (use_crf and hard_constraint) else None
    start_constraint = build_start_constraint_mask(label_scheme) if (use_crf and hard_constraint) else None
    model = CRFClassifier(
        encoder=encoder,
        hidden_size=encoder.hidden_size,
        num_tags=len(label_scheme["labels"]),
        dropout=dropout,
        use_crf=use_crf,
        constraint=constraint,
        start_constraint=start_constraint,
    )
    return model


@torch.no_grad()
def predict_char_tags(model, loader, id2label: dict, device: torch.device):
    """
    把子词级预测还原成「每个汉字一个标签」。

    返回：
      char_tags: list[list[str]]  长度 = 该句最后一个 token 覆盖到的字符数
      info: {
        "token_tags":   list[list[str]]  token 级路径（只含真实 token，去 PAD），
        "scores":       list[float]      句级置信度（CRF 为路径平均对数概率的 exp；Linear 为平均最大概率），
        "paths":        list[list[int]]  原始标签 id
      }

    注意：原文里的空格/换行不产生 token，字符级标签在这些位置没有值；
    所以字符级非法率会有一个由 tokenizer 造成的「假非法」地板，必须和 token 级分开报。
    """
    model.eval()
    char_tags, token_tags, scores, paths_all = [], [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        word_ids = batch["word_ids"]

        if model.use_crf:
            paths, seq_scores = model.predict(
                input_ids, attention_mask, token_type_ids, return_scores=True
            )
        else:
            paths = model.predict(input_ids, attention_mask, token_type_ids)
            # 线性头的置信度：每个 token 最大概率的算术平均（不要连乘）
            emit = model.emissions(input_ids, attention_mask, token_type_ids)
            seq_scores = []
            for b in range(emit.shape[0]):
                n = int(attention_mask[b].sum().item())
                seq_scores.append(float(torch.softmax(emit[b, :n], dim=-1).max(dim=-1).values.mean()))

        for b, path in enumerate(paths):
            wid_row = word_ids[b].tolist()
            tok_tags = [id2label[int(t)] for t in path]
            token_tags.append(tok_tags)
            paths_all.append([int(t) for t in path])
            scores.append(float(seq_scores[b]))

            # 关键：必须按 word_id **原位**写回，不能用 append。
            # 空白字符不产生 token，但 word_ids 里仍占一个位置；append 会让空格之后的
            # 所有标签整体左移，dev 上 trigger/arg 会凭空掉 30+ 个点。
            max_wid = max(wid_row) if wid_row else -1
            n_chars = max_wid + 1 if max_wid >= 0 else 0
            tags = ["O"] * n_chars
            prev = None
            for pos, wid in enumerate(wid_row):
                if wid < 0 or wid >= n_chars:
                    prev = wid
                    continue
                if wid != prev:
                    tags[wid] = tok_tags[pos] if pos < len(tok_tags) else "O"
                    prev = wid
            char_tags.append(tags)

    return char_tags, {"token_tags": token_tags, "scores": scores, "paths": paths_all}


def build_optimizer(model: CRFClassifier, lr: float, head_lr_mult: float = 5.0,
                    weight_decay: float = 0.01):
    """
    分层学习率：预训练骨干用 lr，任务头（Linear + CRF）用 lr*mult。
    偏差 / LayerNorm / CRF 的转移参数不做 weight decay（常规做法，避免把偏置压成 0）。
    """
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    head_prefixes = ("classifier.", "crf.")
    groups = {
        "backbone_decay": {"params": [], "lr": lr, "weight_decay": weight_decay},
        "backbone_nodecay": {"params": [], "lr": lr, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": lr * head_lr_mult, "weight_decay": weight_decay},
        "head_nodecay": {"params": [], "lr": lr * head_lr_mult, "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith(head_prefixes)
        is_nodecay = any(k in name for k in no_decay) or name.startswith("crf.")
        key = ("head" if is_head else "backbone") + ("_nodecay" if is_nodecay else "_decay")
        groups[key]["params"].append(p)
    param_groups = [g for g in groups.values() if g["params"]]
    return torch.optim.AdamW(param_groups)
