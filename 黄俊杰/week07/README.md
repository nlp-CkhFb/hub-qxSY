# week07 作业：DuEE-Fin 事件抽取方法对比

- 任务：一句公告 → 事件类型 + 触发词 + 论元；13 类，从 DuEE-Fin 分层抽 1500 / 400 条（seed=42）
- 数据：过滤后 train 1494 / dev 397（只保留触发词在前 192 字内的样本）；两种口径都在这 397 条上评测
- 完整事件层：复用课程 `common/`（`event_data` / `evaluate_events` / `eval_utils`）和 `rule/run_rule.py`，规则 L0
- 触发词层：`compare.py` 对比规则 / BiLSTM / rbt3 类型+规则定位三种方法
- 说明：BERT 全事件在 CPU 上 1 epoch 没收敛（inner span F1 = 0），没有把不可信的数字写进表里；完整事件只跑规则 L0

完整事件（397 条，课程 `evaluate_events` 口径）：

| 方法 | Trigger F1 | Argument F1 | Event F1 | Type Macro-F1 |
|---|---:|---:|---:|---:|
| 规则 / 词典 + 正则（L0） | 0.6263 | 0.1413 | 0.0076 | 0.7891 |

触发词层（同一批 397 条，span+类型完全匹配）：

| 方法 | Precision | Recall | F1 | Type Acc | 耗时 |
|---|---:|---:|---:|---:|---:|
| 规则 / 词典 | **0.7758** | **0.7758** | **0.7758** | 0.8413 | 0.0 s |
| BiLSTM（无预训练） | 0.6045 | 0.6045 | 0.6045 | 0.8388 | 89 s |
| rbt3 类型 + 规则定位 | 0.7280 | 0.7280 | 0.7280 | **0.9194** | 101 s |

- 结论：完整事件层规则只有 event F1 0.0076，论元/事件级仍是瓶颈；触发词层规则词典最强，rbt3 把类型准确率提到 0.9194，BiLSTM 在 1500 句上欠拟合。
- 运行：
  - 完整事件规则：`python rule/run_rule.py --data_dir data/processed --name hw_rule`
  - 触发词对比：`python compare.py`
- 文件：`compare.py`、`common/`、`rule/run_rule.py`、`data/processed/`、`outputs/`、`results.csv`
