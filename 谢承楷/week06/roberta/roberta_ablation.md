# 消融实验 · RoBERTa 底座（chinese-roberta-wwm-ext）

口径：同一份训练集、同一批无重叠验证样本（8,901 条）、固定 seed=42、3 epoch。每组只改一个旋钮；相对基线单位为 Macro-F1 百分点。

| # | 配置 | Macro-F1 | 相对基线 |
|---|---|---|---|
| exp1 | CLS 池化（基线） | 0.5468 | — |
| exp2 | Mean 池化 | 0.5492 | +0.2 |
| exp3 | Max 池化 | 0.5443 | -0.2 |
| exp4 | CLS + Mean 拼接 | 0.5416 | -0.5 |
| exp5 | 学习率 3e-5 | 0.5444 | -0.2 |
| exp6 | max_length 32 | 0.5452 | -0.2 |
| exp7 | + FGM 对抗训练 | 0.5464 | -0.0 |
| exp8 | + Label Smoothing | 0.5397 | -0.7 |
| exp9 | Focal Loss (γ=2) | 0.5437 | -0.3 |

基线（exp1）= 0.5468。
