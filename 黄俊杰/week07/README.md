# week07 作业：DuEE-Fin 事件抽取方法对比

- 任务：一句公告 → 事件类型 + 触发词 + 论元；13 类，从 DuEE-Fin 分层抽 1500 / 400 条（seed=42）
- 数据：过滤后 train 1494 / dev 397（只保留触发词在前 192 字内的样本）；所有方法都在这 397 条上评测
- 完整事件层：复用课程 `common/`（`event_data` / `evaluate_events` / `eval_utils`）和 `rule/run_rule.py`
  - 规则 / 词典 + 正则（L0）
  - **rbt3 全事件（Linear / CRF / CRF+硬约束）**：`bert/train.py`，149 个 joint 标签，GPU 训练
- 触发词层：`compare.py` 对比规则 / BiLSTM / rbt3 类型+规则定位三种方法
- 环境：RTX 4060 8GB，torch 2.6.0+cu124，transformers 4.55.0；`hfl/rbt3`（3 层，38.6M 参数）

完整事件（397 条，课程 `evaluate_events` 口径）：

| 方法 | 解码器 | Trigger F1 | Argument F1 | Event F1 | Type Macro-F1 | 训练耗时 |
|---|---|---:|---:|---:|---:|---:|
| 规则 / 词典 + 正则（L0） | — | 0.6263 | 0.1413 | 0.0076 | **0.7891** | 0 s（CPU） |
| rbt3 全事件 | Linear | 0.6931 | 0.3307 | 0.0368 | 0.6136 | 91 s |
| rbt3 全事件 | CRF（软约束） | 0.7209 | 0.3432 | 0.0337 | 0.6271 | 160 s |
| rbt3 全事件 | CRF + 硬约束 | **0.8070** | **0.4977** | **0.0891** | 0.7142 | 158 s |

非法 BIO 统计（dev 397 条，字符级口径，模型原始输出）：

| 解码器 | 含非法序列 | 非法开头 | 非法转移 |
|---|---:|---:|---:|
| Linear | 371（93.5%） | 3 | 1780 |
| CRF（软约束） | 370（93.2%） | 3 | 1703 |
| CRF + 硬约束 | 40（10.1%） | 0 | 79 |

- 硬约束后 **token 级非法转移为 0**；剩下的字符级 79 处是「空白字符不产生 token、被强制标 O」造成的
  假非法（`O → I-*`），见 `common/models.py` 里 `predict_char_tags` 的注释。软约束 CRF 只能降低、
  不能消除非法序列，这条对比正是课程「CRF 为什么要加硬约束」的量化证据。

触发词层（同一批 397 条，span+类型完全匹配）：

| 方法 | Precision | Recall | F1 | Type Acc | 耗时 |
|---|---:|---:|---:|---:|---:|
| 规则 / 词典 | **0.7758** | **0.7758** | **0.7758** | 0.8413 | 0.0 s |
| BiLSTM（无预训练） | 0.6045 | 0.6045 | 0.6045 | 0.8388 | 89 s |
| rbt3 类型 + 规则定位 | 0.7280 | 0.7280 | 0.7280 | **0.9194** | 101 s |

## 结论

1. **BERT 全事件明显强于规则**：Argument F1 0.1413 → 0.3432（+0.20），Event F1 0.0076 → 0.0337，
   Trigger F1 0.6263 → 0.7209；加硬约束后更高（0.4977 / 0.0891 / 0.8070）。规则的问题是论元狂喷
   （arg_spurious 56.7%），BERT 学到的论元边界准得多。
2. **但 Type Macro-F1 反而低于规则**（0.61~0.71 vs 0.7891）。逐类看：BERT 在常见类型上很强
   （中标 0.947、亏损 0.978、股份回购 0.973、企业收购 0.959），但**稀有类型直接归零**
   —— 股东增持（dev 13 条）、被约谈（5 条）recall = 0，高管变动 recall 只有 0.048；
   规则的词典是全覆盖。这正是数据「类别不均衡 18.5×」在小样本下的放大效应，
   也是 macro-F1 和 micro-F1 会给出相反结论的典型例子。
3. **3 epoch 不够**：本子集 1,345 条训练句、batch 32 只有 42 步/epoch，3 epoch = 129 步。
   实测 3 epoch 时模型退化成全 `O`（内层验证 span F1 = 0，dev 397 句一句触发词都没抽到），
   指标全 0，**不能当成正式结果**（`outputs/metrics/disc/hw_rbt3_crf.json` 是这次失败尝试的存档）。
   改成 10 epoch（430 步）后正常收敛，最优 epoch 由内层验证集选出（CRF 取 epoch 9），dev 全程不参与选点。
4. 硬约束不只是「消非法」：指标也一起涨（Argument 0.3432 → 0.4977，Event 0.0337 → 0.0891），
   说明小数据 + 149 标签时，把搜索空间收紧对学习本身也有帮助。

## 运行

```bash
cd 黄俊杰/week07

# 1) 规则完整事件（CPU，秒级）
python rule/run_rule.py --data_dir data/processed --name hw_rule

# 2) 触发词层三方法对比（CPU 也能跑，rbt3 首次下载约 150MB）
python compare.py

# 3) rbt3 全事件 + CRF（GPU）
python bert/train.py --data_dir data/processed --bert_path hfl/rbt3 \
  --decoder crf --max_len 192 --epochs 10 --batch_size 32 --device cuda \
  --run_name hw_rbt3_crf_e10

# 4) rbt3 全事件 + Linear（对照）
python bert/train.py --data_dir data/processed --bert_path hfl/rbt3 \
  --decoder linear --max_len 192 --epochs 10 --batch_size 32 --device cuda \
  --run_name hw_rbt3_linear_e10

# 5) rbt3 全事件 + CRF + 硬约束
python bert/train.py --data_dir data/processed --bert_path hfl/rbt3 \
  --decoder crf --hard_constraint --max_len 192 --epochs 10 --batch_size 32 --device cuda \
  --run_name hw_rbt3_crf_hard_e10
```

- 指标：`outputs/metrics/disc/<run_name>.json`；预测：`outputs/disc/pred_<run_name>.jsonl`
- 训练曲线：`outputs/logs/train_<run_name>.json`
- 权重 `outputs/checkpoints/*.pt` 约 150MB，**不提交**，脚本会重新训练

## 文件

`compare.py`、`common/`、`rule/run_rule.py`、`bert/train.py`、`GPU_RUN.md`、`data/processed/`、
`outputs/metrics/disc/`、`outputs/disc/`、`results.csv`

## 已知问题

- dev 里 397 条全部是「有事件」的样本（过滤后），所以本作业**没有做事件检测（有无事件）这一层**，
  77.9% 无事件的句子在原始 `data/dev.jsonl` 里，未参与训练/评测。
- `data/processed` 是 1,494 / 397（过滤版），不是 1,500 / 400；规则和 BERT 用同一批样本，口径一致。
- 论元层仍是瓶颈：最好的 Event exact F1 只有 0.0891，论元角色长尾（61 个角色、92 个槽位）远没吃透。
