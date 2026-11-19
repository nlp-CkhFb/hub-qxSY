# week06 作业：TNEWS 新闻分类方法对比

- 任务：15 类新闻标题分类；从课程 TNEWS 分层抽 3000 条训练、500 条验证（seed=42）
- 数据：验证集先剔除与全量训练集重复的 1,099 条（去重后 8,901 条）再抽，避免词面泄漏
- 改动：预训练模型从课程的 `bert-base-chinese`（102M）换成 3 层 `hfl/rbt3`（38M），CPU 可跑
- 方法：规则 / TF-IDF+LinearSVC / rbt3 冻结特征+SVM / rbt3 微调（max_len 64，batch 16，lr 2e-5，2 epoch，类别加权）
- 结果（同一批 500 条，seed=42）：

| 方法 | Accuracy | Macro-F1 | 耗时 |
|---|---:|---:|---:|
| 规则 / 关键词 | 0.1300 | 0.0646 | 0.0 s |
| TF-IDF + LinearSVC | 0.3980 | 0.4033 | 0.6 s |
| rbt3 冻结 + SVM | 0.4000 | 0.3765 | 59 s |
| rbt3 微调 | **0.4580** | **0.4223** | 396 s |

- 结论：rbt3 微调效果最好（Macro-F1 0.4223，比 TF-IDF 高 1.9 个点），但耗时约 400 秒；TF-IDF 0.6 秒就到 0.4033，短文本小样本仍是很强的基线。冻结特征只有 0.3765，说明预训练模型不微调不一定比 TF-IDF 好。
- 说明：单 seed、3,000 条子集，数字只用于方法横向对比；规则只匹配类别名，命中率 5.8%。
- 运行：`python compare.py`；只跑基线 `python compare.py --methods rule,tfidf`。首次运行会下载 `hfl/rbt3`（约 150MB）。
- 文件：`compare.py`、`data/train.json`、`data/val.json`、`results.csv`
