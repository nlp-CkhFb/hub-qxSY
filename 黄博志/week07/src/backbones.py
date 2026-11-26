"""
预训练骨干的定位与加载。

本实验只改一个变量：把 BERT-base-chinese 换成同为 12 层 / 768 隐层 / 102M 参数的
RoBERTa-base-chinese（ModelScope damo/nlp_roberta_backbone_base_std，week06 已下载到本地）。
两者容量对齐，因此「RoBERTa 比 BERT 高/低多少」可以解释成预训练目标的差异，
而不是模型大小的差异。

刻意不联网下载：路径写死在本机绝对路径 + 可用环境变量覆盖，换机器时改这里即可。
"""

import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

# 本机已有的两个中文 base 权重（绝对路径）
BACKBONES = {
    "bert-base-chinese": {
        "display": "BERT-base-chinese",
        "family": "bert",
        "path": r"D:\八斗\week4\第四周 语言模型\bert-base-chinese",
        "params_m": 102.4,
        "note": "Google 中文 BERT-base，MLM 预训练，week4 课程材料自带",
    },
    "roberta-base-chinese": {
        "display": "RoBERTa-base-chinese",
        "family": "roberta",
        "path": r"C:\Users\hbz\Desktop\work_by_git\hub-qxSY\黄博志\week06\nlp_roberta_backbone_base_std",
        "params_m": 102.4,
        "note": "ModelScope damo/nlp_roberta_backbone_base_std，RoBERTa 预训练目标，week06 已用于 TNEWS 分类对比",
    },
}

ALIASES = {
    "bert": "bert-base-chinese",
    "roberta": "roberta-base-chinese",
    "roberta-base": "roberta-base-chinese",
    "bert-base": "bert-base-chinese",
}


def resolve_backbone(spec: str) -> dict:
    """把 --backbone 的短名 / 绝对路径解析成一个描述 dict。"""
    spec = (spec or "").strip()
    key = ALIASES.get(spec, spec)
    if key in BACKBONES:
        info = dict(BACKBONES[key])
        info["key"] = key
        path = Path(info["path"])
        if not path.exists():
            raise FileNotFoundError(
                f"找不到 {key} 的本地权重：{path}\n"
                f"请修改 src/backbones.py 里的 BACKBONES 路径。"
            )
        info["path"] = str(path)
        return info

    # 也允许直接传一个本地目录（方便以后换模型）
    path = Path(spec)
    if path.is_dir() and (path / "config.json").exists():
        cfg = AutoConfig.from_pretrained(str(path))
        return {
            "key": path.name,
            "display": path.name,
            "family": getattr(cfg, "model_type", "unknown"),
            "path": str(path),
            "params_m": round(sum(p.numel() for p in AutoModel.from_pretrained(str(path)).parameters()) / 1e6, 1),
            "note": "用户指定的本地目录",
        }
    raise ValueError(f"未知骨干：{spec}（可用：{', '.join(BACKBONES)}，或传本地模型目录）")


def load_tokenizer(model_path: str):
    """优先 fast tokenizer；个别 ModelScope 目录没有 tokenizer.json 时也能从 vocab.txt 构造。"""
    try:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        from transformers import BertTokenizerFast
        tok = BertTokenizerFast.from_pretrained(model_path)
    if not tok.is_fast:
        raise RuntimeError(f"{model_path} 只加载到了 slow tokenizer，无法用 word_ids() 做子词对齐")
    return tok


class TransformerEncoder(torch.nn.Module):
    """
    统一封装 BERT / RoBERTa 的编码器，输出 (B, L, H)。

    - num_layers < 12 时保留预训练权重、只截断 encoder.layer 的前 N 层
      （不是 BertConfig(num_hidden_layers=N) 随机初始化 —— 那会把「层数」和「预训练」混在一起）。
    - RoBERTa 的 forward 不一定接受 token_type_ids；这里先探测签名，避免换骨干时静默报错。
    """

    def __init__(self, model_path: str, num_layers: int = 0, dropout: float | None = None):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_path)
        self.model = AutoModel.from_config(cfg)
        self._load_local_weights(model_path)
        self.hidden_size = int(cfg.hidden_size)
        self.num_layers_total = len(self.model.encoder.layer)

        if dropout is not None:
            cfg.hidden_dropout_prob = dropout
            cfg.attention_probs_dropout_prob = dropout

        if num_layers and 0 < num_layers < self.num_layers_total:
            self.model.encoder.layer = torch.nn.ModuleList(
                list(self.model.encoder.layer)[:num_layers]
            )
            cfg.num_hidden_layers = num_layers
            self.num_layers_used = num_layers
        else:
            self.num_layers_used = self.num_layers_total

        import inspect
        self.accepts_token_type_ids = "token_type_ids" in inspect.signature(
            self.model.forward
        ).parameters

    def _load_local_weights(self, model_path: str):
        """
        手动加载本地权重，而不是 AutoModel.from_pretrained。

        原因：week4 / week06 这两个本地 checkpoint 的 LayerNorm 参数名是
        `.gamma` / `.beta`（ModelScope 旧实现），transformers 5.x 只认
        `.weight` / `.bias`。直接 from_pretrained 会把这 26×2 个 LayerNorm 仿射
        参数全部丢掉并用 1/0 初始化：
          - BERT 的 gamma 均值 0.963，接近 1，掉参数影响小；
          - RoBERTa 的 gamma 均值 0.609、范围 0.02~7.23，掉参数后预训练表示被严重破坏，
            3 epoch 内明显欠拟合（dev 论元 F1 0.32 vs BERT 0.55）。
        这里做键名映射（去掉 `bert.` 前缀 + gamma/beta→weight/bias），并断言
        没有遗漏任何 backbone 参数。
        """
        bin_path = Path(model_path) / "pytorch_model.bin"
        if not bin_path.exists():
            # 标准 HF 目录：交给 from_pretrained 即可
            loaded = AutoModel.from_pretrained(model_path)
            self.model.load_state_dict(loaded.state_dict(), strict=True)
            return

        raw = torch.load(str(bin_path), map_location="cpu", weights_only=False)
        converted = {}
        for key, value in raw.items():
            if key.startswith("bert."):
                key = key[5:]
            if key.endswith(".gamma"):
                key = key[:-6] + ".weight"
            elif key.endswith(".beta"):
                key = key[:-5] + ".bias"
            converted[key] = value

        target = self.model.state_dict()
        filtered = {k: v for k, v in converted.items()
                    if k in target and tuple(target[k].shape) == tuple(v.shape)}
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        missing_backbone = [k for k in missing if not k.startswith("cls.")]
        if missing_backbone:
            raise RuntimeError(f"{model_path} 权重不完整，缺失 {len(missing_backbone)} 个 backbone 参数："
                               f"{missing_backbone[:5]}")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.accepts_token_type_ids and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.model(**kwargs)
        # transformers 5.x 的 return_dict 默认值与 4.x 不同，可能返回 tuple
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        scale = float(os.environ.get("WEEK07_EMISSION_SCALE", "1.0"))
        return hidden * scale if scale != 1.0 else hidden
