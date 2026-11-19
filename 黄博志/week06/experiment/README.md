# 换预训练模型对比实验：BERT-base vs RoBERTa-base

本目录是第六周分类作业的“只换预训练模型”版本。数据、任务、标签体系都不变，仍然是
TNEWS 15 类中文新闻短文本分类；唯一变化的变量是**预训练骨干**。

## 1. 要回答的问题

1. 在完全相同的训练/评测口径下，`bert-base-chinese` 换成
   `nlp_roberta_backbone_base_std`，微调后的 Macro-F1 会差多少？
2. 如果不微调、只把 `[CLS]` 当句向量训练逻辑回归，换骨干还有没有差别？
3. 换模型带来的收益，值不值它多花的时间/显存/权重体积？

## 2. 两个骨干

| 骨干 | 来源 | 结构 | 参数量 | 权重 |
| --- | --- | --- | --- | --- |
| BERT-base-chinese | 课程提供 / 本地 | 12 层、768 隐藏、12 头 | 约 102M | 393 MB |
| RoBERTa-base (ModelScope) | `nlp_roberta_backbone_base_std` | 12 层、768 隐藏、12 头 | 约 102M | 411 MB |

两个骨干容量几乎一样，所以差异主要来自**预训练数据与预训练目标**，而不是“模型更大”。
注意：提供的 RoBERTa 是 ModelScope 旧格式，权重 key 带 `bert.` 前缀，LayerNorm 参数叫
`.gamma` / `.beta`；`models.py` 里做了显式转换，并断言没有 missing key。

## 3. 数据与切分

TNEWS 原始数据：53,360 条训练、10,000 条验证，15 类。

先做泄漏清洗，再做固定切分：

- 训练集内部按整句完全去重：53,360 → 49,726；
- 验证集删除与训练集重复的句子：10,000 → 8,901；
- 从训练集分层抽 7,000 条训练、1,000 条开发集；
- 从去重后的验证集分层抽 2,000 条测试集。

固定 `seed=42`，切分结果缓存在 `outputs/splits.json`。所有方法共用同一份切分。

> 为什么不用全量：本机只有 CPU（16 线程），BERT-base 全量 5 万条一个 epoch 约 4 小时。
> 7k/1k/2k 是在“统计上能看出差异”和“CPU 可跑完”之间的折中。结论应表述为
> “在该数据规模下的趋势”，而不是“全量 TNEWS 的 SOTA”。

## 4. 方法矩阵

| 方法 | 说明 | 训练成本 |
| --- | --- | --- |
| `rule` | 15 类人工关键词，命中数取 argmax | 0 |
| `tfidf_svc` | char 2–4 gram TF-IDF + LinearSVC | 秒级 |
| `tfidf_nb` | 同特征 + MultinomialNB | 秒级 |
| `bert_frozen` | BERT 冻结，`[CLS]` + LogisticRegression | 约 5–10 分钟 |
| `roberta_frozen` | RoBERTa 冻结，`[CLS]` + LogisticRegression | 约 5–10 分钟 |
| `bert_ft` | BERT 全量微调，2 epoch | 约 35–45 分钟 |
| `roberta_ft` | RoBERTa 全量微调，2 epoch | 约 35–45 分钟 |

微调的控制变量：`max_len=40`、`batch=32`、`epochs=2`、`lr=2e-5`、
`weight_decay=0.01`、`warmup=10%`、`grad_clip=1.0`、class weights、`seed=42`。

## 5. 目录结构

```
week06/
├─ data/
│  ├─ train.json          # 从课程 code/data 复制
│  └─ val.json
├─ nlp_roberta_backbone_base_std/   # 你提供的 RoBERTa 权重
├─ experiment/
│  ├─ config.py           # 路径、超参、标签
│  ├─ data.py             # 去重 + 固定切分
│  ├─ metrics.py          # Macro-F1 / 每类 P-R-F1 / 混淆矩阵 / ECE / McNemar
│  ├─ models.py           # 两个骨干加载 + RoBERTa 权重转换 + [CLS] 编码
│  ├─ baselines.py        # 规则 + TF-IDF
│  ├─ embed_lr.py         # 冻结特征 + LR
│  ├─ finetune.py         # 微调
│  ├─ run_all.py          # 一键跑
│  └─ report.py           # 汇总报告
└─ outputs/
   ├─ splits.json
   ├─ summary.md / summary.csv
   ├─ errors.md
   ├─ confusion_*.png
   ├─ per_class_f1.png
   └─ <method>/metrics.json, predictions.json
```

## 6. 运行

```powershell
# 正式实验：约 1.5–2 小时 CPU，建议后台跑
py -3.11 run_all.py --methods all

# 只重跑某几个方法
py -3.11 run_all.py --methods bert_ft,roberta_ft --force

# 冒烟测试：小样本、5 步，约 2–3 分钟
py -3.11 run_all.py --smoke --methods all --force
```

依赖（本机 Python 3.11 已满足）：

```powershell
py -3.11 -m pip install -r requirements.txt
```

如果换机器，路径可以用环境变量覆盖：

```powershell
$env:BERT_DIR = "D:\...\bert-base-chinese"
$env:ROBERTA_DIR = "D:\...\nlp_roberta_backbone_base_std"
$env:TNEWS_DATA_DIR = "D:\...\code\data"
```

## 7. 输出怎么读

- `outputs/summary.md`：主报告，按 Macro-F1 排序的主表、核心结论、每类 F1、
  混淆矩阵、Top 混淆对、类别 ID 映射。
- `outputs/summary.csv`：同样的主表，Excel 可直接打开。
- `outputs/errors.md`：两个微调模型按置信度排序的错例。
- `outputs/<method>/predictions.json`：逐条预测，含真实标签、预测标签、概率。
- `outputs/<method>/metrics.json`：统一评测结果。

指标口径：

- **Macro-F1**：15 类等权，主指标；类别不平衡时比 Accuracy 更能暴露问题。
- **Accuracy**：整体正确率，受大类影响。
- **Weighted-F1**：按 support 加权，和 Accuracy 接近。
- **ECE**：有概率输出的方法才计算，衡量置信度校准；规则/LinearSVC 为 `-`。
- **McNemar**：只在两个模型逐条预测可配对时计算，用来判断差异是否显著。

## 8. 预期与注意事项

- 这是**单 seed、2,000 条测试集**的实验；Macro-F1 差 1 个点以内、或 McNemar
  `p >= 0.05` 时，建议说“趋势”，不要说“显著更好”。
- TNEWS 类别很不平衡（7k 训练集里“证券”只有 35 条，“科技”有 766 条），
  每类 F1 的方差会比较大；主指标用 Macro-F1，但也要看每类召回。
- 冻结 `[CLS]` 对 BERT 比较友好（BERT 有 NSP，[CLS] 学过句子级目标）；
  RoBERTa 类模型没有 NSP，冻结 `[CLS]` 可能明显弱于 BERT。这个差异本身就是一个
  值得写进报告的发现，不要只看微调结果。
- 如果之后要加数据集对比，`THUCNewsTitle`（10 类、50k/5k/10k、标题平均 18.9 字）
  是 CPU 友好的候选；但它和 TNEWS 标签体系不同，指标不能直接横比，只能讨论趋势。
