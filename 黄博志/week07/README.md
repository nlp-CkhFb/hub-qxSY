# Week 07 信息抽取：BERT vs RoBERTa，Linear / CRF / 解码期硬约束

本目录是第七周抽取任务的独立复现与扩展实验。目标有两个：

1. **换骨干**：把课程基线里的 `bert-base-chinese` 换成同为 12 层 / 768 隐层 / 102M 参数的 `RoBERTa-base-chinese`（ModelScope `damo/nlp_roberta_backbone_base_std`，week06 已下载到本地），在完全相同的数据、标签、训练预算和评测脚本下比较。
2. **换方法**：在同一骨干上比较 **Linear 解码**、**CRF 解码**、以及 **CRF + 解码期硬约束**，并给出配对显著性检验与成本对比。

> 所有数字均由 `experiments/summarize.py` 从 `outputs/metrics/disc/*.json` 自动汇总；叙述性结论以本文件为准，明细表见 `report/results.md`，显著性原始结果见 `report/significance.json`。

---

## 1. 数据与评测口径

- 数据：DuEE-Fin 派生，一句一事件；**13 个事件类型 / 92 个槽位 / 61 个去重角色**。
- 规模：train **5,747** 句，dev **930** 句；train 内再切 10% 做内层验证选 epoch，**dev 只用于最终评测**。
- 标签：joint BIO，**149 个标签**（13 类型 × 触发词 + 论元角色）。
- 评测：直接复用课程的 `src/evaluate_events.py`（`row_key = id|sent_id`；trigger 文本完全匹配；论元按 `(role, text)` Counter 一对一；event 要求类型、触发词、论元全部匹配）。所有方法输出同一种 JSONL，口径与课程 `common/evaluate_events.py` 一致。
- 数据事实：extract 平均 164 字，p50 161，p90 273，p99 393，max 797；`max_len=256` 截断 train 12.1% / dev 13.1%（122/930）。

## 2. 实验设计

### 2.1 骨干

| 名称 | 本地路径 | 结构 | 参数量 |
|---|---|---|---:|
| BERT-base-chinese | `D:\八斗\week4\第四周 语言模型\bert-base-chinese` | 12 层 / 768 / 12 头 / vocab 21128 | 102.4M |
| RoBERTa-base-chinese | `C:\Users\hbz\Desktop\work_by_git\hub-qxSY\黄博志\week06\nlp_roberta_backbone_base_std` | 12 层 / 768 / 12 头 / vocab 21128 | 102.4M |

RoBERTa 权重来自 ModelScope `damo/nlp_roberta_backbone_base_std`。其 `config.json` 的 `model_type` 是 `bert`（StructBERT 结构），因此用 `AutoModel` 按 BERT 结构加载；两份 `vocab.txt` 仅 2 行差异，中文均逐字切分。需要说明的是，该 checkpoint 的 MLM 头有转换瑕疵（`cls.predicitions.bias` 拼写错误、缺少 `cls.predictions.decoder`），但 **backbone 的 12 层参数完整**，本实验只使用 backbone。

### 2.2 方法

1. **Linear**：`last_hidden_state` → `Linear(768, 149)`，逐 token 分类，交叉熵。
2. **CRF**：同一 encoder → 线性 emission → 手写 CRF（`src/crf.py`，含转移矩阵、start/end、维特比）。训练用负对数似然，解码用维特比。
3. **CRF + 硬约束**：不重新训练，在同一个 CRF checkpoint 上把非法 BIO 转移（如 `O → I-X`、`B-X → I-Y`）和非法开头（`I-X` 不能作为序列首标签）的分数置为 `-1e4` 后重跑维特比。该操作只影响解码，训练成本为 0。

### 2.3 公平性

- 同一份 `data/processed`、同一个 149 标签空间、同一个 `BioDataset`/`collate_bio`。
- 相同训练预算：3 epoch、有效 batch 32（`batch_size=8 × grad_accum=4`）、lr 2e-5、head lr ×5、warmup 10%、max_len 256、seed 42。
- 模型选择只看 train 切出的 10% 内层验证 span F1；dev 不参与任何超参选择。
- 两个骨干使用**同一个** `TransformerEncoder` 和同一套权重加载逻辑。

> 硬件：GTX 1650 Ti 4GB（WDDM）。bs16 会触发共享显存换页（3.66GB、3.13s/step），因此统一用 bs8；AMP 在无 tensor core 的 1650 Ti 上反而慢 3.7×，全程 fp32。

## 3. 主结果（dev 930）

| 方法 | 触发词 F1 | 论元 F1 | 事件 F1 | Δ论元 vs BERT+Linear | Δ事件 | n200 论元 | 训练(s) | 推理(ms/句) | 非法率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT + Linear | 0.8242 | 0.5845 | 0.1868 | +0.00pp | +0.00pp | 0.5917 | 1365 | 61.1 | 64.8% |
| BERT + CRF | 0.8251 | 0.5952 | 0.1782 | +1.07pp | −0.86pp | 0.5958 | 1936 | 81.9 | 64.6% |
| **BERT + CRF + 硬约束** | **0.8441** | **0.6611** | **0.2281** | **+7.66pp** | **+4.13pp** | **0.6692** | 0 | 81.6 | 7.7% |
| RoBERTa + Linear | 0.4866 | 0.2058 | 0.0024 | −37.87pp | −18.44pp | 0.1901 | 1366 | 62.0 | 95.0% |
| RoBERTa + CRF | 0.4743 | 0.2074 | 0.0024 | −37.71pp | −18.44pp | 0.1939 | 1946 | 81.8 | 95.3% |
| RoBERTa + CRF + 硬约束 | 0.4958 | 0.2479 | 0.0024 | −33.66pp | −18.44pp | 0.2307 | 0 | 78.9 | 5.9% |

注：非法率对 Linear/CRF 取 token 级非法序列比例；硬约束行 token 级为 0，表中 7.7% 是字符级（空格不产生 token 造成的“假非法”）。

### 3.1 与课程文件内基线对照

课程基线（`baseline/`，原仓库 `outputs/metrics/disc/summary.md`）：

| 方法 | 指标 | 课程基线 | 本实验 | Δ(pp) |
|---|---|---:|---:|---:|
| BERT + Linear | 触发词 / 论元 / 事件 | 0.8293 / 0.5630 / 0.1564 | 0.8242 / 0.5845 / 0.1868 | −0.51 / **+2.15** / **+3.04** |
| BERT + CRF | 触发词 / 论元 / 事件 | 0.8320 / 0.5815 / 0.1697 | 0.8251 / 0.5952 / 0.1782 | −0.69 / **+1.37** / **+0.85** |
| BERT + CRF + 硬约束 | 触发词 / 论元 / 事件 | 0.8464 / 0.6566 / 0.2177 | 0.8441 / 0.6611 / 0.2281 | −0.23 / **+0.45** / **+1.04** |

三种 BERT 配置的论元 F1 都略高于课程记录（+0.45 ~ +2.15pp），触发词略低（−0.23 ~ −0.69pp）。差异来源：本实验显式恢复了 ModelScope 风格 checkpoint 的 LayerNorm `.gamma/.beta`（见 §4.2）、batch 为 8×4 而非 32、以及随机种子不同。**结论层面与课程一致**：CRF 优于 Linear，硬约束带来最大增幅。

## 4. 关键发现

### 4.1 CRF 的收益集中在论元边界，硬约束才是大头

- BERT Linear → CRF：论元 **+1.07pp**（95% CI [+0.17, +1.99]，bootstrap p=0.019），触发词 +0.09pp（不显著），事件 −0.86pp（CI [−2.29, +0.55]，p=0.232，不显著）。
- BERT CRF → 硬约束：触发词 **+1.90pp**、论元 **+6.59pp**、事件 **+4.99pp**，bootstrap p=0.0002，McNemar p<0.0001。三个指标同时显著提升，且训练成本为 0。
- 机制与课程 NOTES 一致：软约束下模型会解出 `I-X` 开头或 `B-X → I-Y` 的非法片段，硬约束直接把这些路径的分数置为 `-1e4`，从而修正边界；token 级非法率从 62~65% 降到 0。
- 与课程不同的是，本实验 BERT CRF 的事件 F1 略低于 Linear（课程为 +1.3pp）。差距只有 0.86pp 且不显著；在本实验的加载/种子下，CRF 的边界收益没有转化为事件完全匹配收益。

### 4.2 RoBERTa 崩溃的根因：LayerNorm 参数与激活尺度

这是本次复现最重要的工程发现。

1. **`.gamma/.beta` 被 transformers 5.x 丢弃**：两份本地 checkpoint 的 LayerNorm 参数名是 ModelScope 旧格式 `.gamma/.beta`，而 transformers 5.x 只认 `.weight/.bias`。直接 `AutoModel.from_pretrained` 会丢掉 **26 组 × 2 个**仿射参数，并用 1/0 初始化。
2. 本实验在 `src/backbones.py::_load_local_weights` 中显式做了 `bert.` 前缀剥离、`.gamma → .weight`、`.beta → .bias` 的重映射，并断言 backbone 无缺失。
3. 恢复后，两个骨干的 LayerNorm 尺度差异很大：
   - BERT：`LayerNorm.weight` 均值 **0.963**，隐状态 std **0.825**，norm **22.85**；
   - RoBERTa：`LayerNorm.weight` 均值 **0.609**（范围 0.02~7.23），隐状态 std **0.383**，norm **10.63**。
4. RoBERTa 的隐状态整体被压缩到约 BERT 的一半，`Linear(768,149)` 的初始 logits 更小，3 epoch 预算内收敛极慢：Linear 训练 loss 从 5.0 只降到 1.92（BERT 降到 1.32），内层验证 span F1 仅 0.0024，dev 论元 0.2058，token 非法率 95%。
5. **尺度校准诊断**：把 encoder 输出乘以 **2.0**（约等于 0.825/0.383，对齐到 BERT 的激活尺度）后重跑，同一 3 epoch 预算下（完整设置与显著性见 `report/scale_study.md`）：

| RoBERTa 配置 | 触发词 | 论元 | 事件 | 内层验证 spanF1 | 训练(s) |
|---|---:|---:|---:|---:|---:|
| Linear，原始尺度 | 0.4866 | 0.2058 | 0.0024 | 0.2699 | 1366 |
| Linear，×2 尺度 | 0.7654 | 0.4263 | 0.0853 | 0.4872 | 1460 |
| CRF，×2 尺度 | 0.7912 | 0.4762 | 0.1089 | 0.5318 | 1946 |
| CRF + 硬约束，×2 尺度 | **0.8207** | **0.6208** | **0.1781** | — | 0 |

尺度校准后 RoBERTa 的论元 F1 从 0.2058 提升到 0.4762（CRF），硬约束再到 0.6208，提升幅度 **+41.5pp**；但同方法下仍低于 BERT：CRF 低 3.38 / 11.90 / 6.93pp（触发/论元/事件，p≤0.0002），硬约束低 2.33 / 4.02 / 5.00pp（p≤0.006）。**校准能救回大部分，但没有追平。**

6. 尺度校准后，CRF 相对 Linear 的收益也恢复且显著：触发 +2.58pp、论元 +4.99pp、事件 +2.36pp（p≤0.0024）；硬约束再把 CRF 的论元从 0.4762 提到 0.6208（+14.5pp）、事件从 0.1089 提到 0.1781（+6.9pp）。说明 **方法结论（CRF/硬约束有效）在 RoBERTa 上同样成立**，只是绝对水平受骨干预训练质量限制。

> 加载方式消融（单次运行，日志见 `outputs/logs/`）：若不做 `.gamma/.beta` 重映射、按 `from_pretrained` 使用单位权重，BERT Linear 论元 0.5538、RoBERTa Linear 论元 0.3207；显式恢复后分别变为 0.5845 / 0.2058。RoBERTa 恢复 gamma 后反而更低，是因为它的 gamma 均值只有 0.609，把本已偏小的激活压得更小，3 epoch 内更难收敛——这正是后文 ×2 尺度校准的动机，也说明 **“正确加载”与“同等优化预算下可训练”是两件事**。

### 4.3 分类型看，RoBERTa 只学会了少数高频类型

`report/per_type.csv` 显示：BERT 在 13 个类型中有 11 个非零，论元 F1 约 0.18~0.85（最好的「股份回购」0.845）；RoBERTa 原始尺度下 **8/13 个类型为 0.000**，只在「股份回购」「中标」「企业收购」「亏损」等高频类型上有 0.17~0.37，属于典型的少数类坍缩。尺度校准后 CRF+硬约束恢复到 11/13 个类型非零（「企业融资」0.63、「公司上市」0.42、「质押」0.60、「企业破产」0.40、「解除质押」0.48），但「高管变动」仅 0.016，「股东增持」「被约谈」仍为 0，说明 3 epoch 下 RoBERTa 对长尾类型的拟合远未完成。

### 4.4 成本

- CRF 训练比 Linear 慢 **1.42×**（1936s vs 1365s），推理慢 **1.34×**（81.9 vs 61.1 ms/句）。
- 硬约束解码与普通 CRF 解码几乎同速（81.6 vs 81.9 ms/句），训练成本为 0，是性价比最高的改动。
- 两个骨干参数量相同（102.4M），单步耗时接近；RoBERTa 的差距来自预训练表示与收敛速度，不是模型规模。

## 5. 复现

### 5.1 环境

```text
Python 3.12.14
torch 2.6.0+cu124
transformers 5.19.0
numpy / scipy / matplotlib / tqdm
GPU: NVIDIA GTX 1650 Ti 4GB
```

安装：

```powershell
pip install -r requirements.txt
```

`torch` 需从 `https://download.pytorch.org/whl/cu124` 安装 CUDA 版；若 pip 走代理卡住，可先用 `curl -L -C -` 下载 wheel 再 `pip install --no-deps`。

### 5.2 命令

```powershell
# 全矩阵：4 次训练 + 2 次硬约束复评（约 2 小时）
python experiments/run_all.py --force

# 单次训练
python src/train.py --backbone bert-base-chinese --decoder crf --run_name bert_crf_256
python src/train.py --backbone roberta-base-chinese --decoder linear --run_name roberta_linear_256

# 硬约束复评（同一 checkpoint，只改解码）
python src/evaluate.py --ckpt outputs/checkpoints/bert_crf_256.pt --name bert_crf_256_hard --hard_constraint

# 汇总 + 配对 bootstrap / McNemar
python experiments/summarize.py --significance
```

RoBERTa 尺度校准诊断：

```powershell
$env:WEEK07_EMISSION_SCALE='2.0'
python src/train.py --backbone roberta-base-chinese --decoder crf --epochs 3 --run_name roberta_crf_scale2_3ep
python src/evaluate.py --ckpt outputs/checkpoints/roberta_crf_scale2_3ep.pt --name roberta_crf_scale2_3ep_hard --hard_constraint
```

### 5.3 目录

```text
week07/
├─ data/processed/          # 课程数据（train/dev extract、schema）
├─ data/gold_n200.jsonl     # 课程固定 n200 子集
├─ src/
│  ├─ backbones.py          # 两个本地骨干的解析、加载、LayerNorm 重映射
│  ├─ event_data.py         # BIO 标签、Dataset、解码、非法统计
│  ├─ evaluate_events.py    # 与课程一致的评测
│  ├─ evaluate.py           # checkpoint → dev 预测/指标
│  ├─ train.py              # 统一训练入口
│  ├─ models.py             # encoder + Linear/CRF + 推理
│  ├─ crf.py                # 手写 CRF
│  └─ eval_utils.py         # 指标落盘、n200 投影、计时
├─ experiments/
│  ├─ run_all.py            # 全矩阵串行
│  ├─ significance.py       # 配对 bootstrap + McNemar
│  └─ summarize.py          # 表格/图/results.md
├─ baseline/                # 课程 BERT 预测与指标（只读对照）
├─ outputs/                 # pred / metrics / logs / checkpoints
└─ report/                  # results.md、summary.csv、per_type.csv、figures/
```

## 6. 局限

1. **3 epoch 预算对 RoBERTa 偏短**：尺度校准后内层验证 F1 仍在上升（0.4872），没有跑到收敛；BERT 在 epoch3 已接近平台。更公平的做法是按收敛早停或给两个骨干各自调 epoch，但这会引入额外超参。
2. **尺度 ×2 是诊断性校准**，不是标准训练流程；它按 BERT/RoBERTa 的隐状态 std 比例设定，未在 dev 上调参。主表仍以原始尺度为准，校准表单独列出。
3. **RoBERTa checkpoint 本身有转换瑕疵**（MLM 头拼写错误、缺 decoder），backbone 参数完整但不能排除预训练质量/转换损失的影响；这可能是它追不平 BERT 的部分原因。
4. 单 seed（42），显著性检验用配对 bootstrap 缓解样本噪声，但没有覆盖训练随机性；严格结论应多 seed 重复。
5. 事件级 F1 对触发词边界和论元完全匹配非常敏感，绝对值不高（BERT 最好 0.2281），适合做相对比较，不宜直接当业务指标。
