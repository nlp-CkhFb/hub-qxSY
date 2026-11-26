# GPU 全事件训练说明（week07）

当前 `data/processed/` 是 1,494 训练 / 397 验证（已过滤触发词在前 192 字内）。
`common/` + `rule/` 已随仓库提交，`bert/train.py` 已改为 `BertTokenizerFast`。

```bash
cd 黄俊杰/week07

# 1) 规则完整事件基线（CPU，秒级）
python rule/run_rule.py --data_dir data/processed --name hw_rule

# 2) rbt3 + CRF，3 epoch（GPU）
python bert/train.py \
  --data_dir data/processed \
  --bert_path hfl/rbt3 \
  --bert_layers 0 \
  --decoder crf \
  --max_len 192 \
  --epochs 3 \
  --batch_size 32 \
  --device cuda \
  --run_name hw_rbt3_crf

# 3) 可选：rbt3 + Linear 对照
python bert/train.py \
  --data_dir data/processed \
  --bert_path hfl/rbt3 \
  --bert_layers 0 \
  --decoder linear \
  --max_len 192 \
  --epochs 3 \
  --batch_size 32 \
  --device cuda \
  --run_name hw_rbt3_linear
```

指标文件：`outputs/metrics/disc/hw_rbt3_crf.json`（或 `hw_rbt3_linear.json`）。
读取：

```python
import json
d = json.load(open("outputs/metrics/disc/hw_rbt3_crf.json", encoding="utf-8"))
print(d["trigger"]["f1"], d["argument"]["f1"], d["event_exact"]["f1"], d["type_macro_f1"])
```

把结果补进 README 的“完整事件”表，然后：

```bash
git add 黄俊杰/week07
git commit -m "黄俊杰-第七周作业：GPU 补充 rbt3+CRF 完整事件结果"
git push fork week07
```

PR #99 会自动更新；不需要新建 PR。
