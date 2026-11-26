# GPU 全事件训练说明（week07）

当前 `data/processed/` 是 1,494 训练 / 397 验证（已过滤触发词在前 192 字内）。
`common/` + `rule/` 已随仓库提交，`bert/train.py` 已改为 `BertTokenizerFast`。

实测环境：RTX 4060 8GB / torch 2.6.0+cu124 / transformers 4.55.0，
`hfl/rbt3`（3 层，38.6M），batch 32 + max_len 192 时峰值显存约 **807 MiB**，
CRF 10 epoch 约 **160 s**（Linear 约 91 s）。

```bash
cd 黄俊杰/week07

# 1) 规则完整事件基线（CPU，秒级）
python rule/run_rule.py --data_dir data/processed --name hw_rule

# 2) rbt3 + CRF，10 epoch（GPU）
python bert/train.py \
  --data_dir data/processed \
  --bert_path hfl/rbt3 \
  --bert_layers 0 \
  --decoder crf \
  --max_len 192 \
  --epochs 10 \
  --batch_size 32 \
  --device cuda \
  --run_name hw_rbt3_crf_e10

# 3) 可选：rbt3 + Linear 对照
python bert/train.py \
  --data_dir data/processed \
  --bert_path hfl/rbt3 \
  --bert_layers 0 \
  --decoder linear \
  --max_len 192 \
  --epochs 10 \
  --batch_size 32 \
  --device cuda \
  --run_name hw_rbt3_linear_e10

# 4) 可选：CRF + 硬约束（非法 BIO 归零的消融）
python bert/train.py \
  --data_dir data/processed \
  --bert_path hfl/rbt3 \
  --bert_layers 0 \
  --decoder crf \
  --hard_constraint \
  --max_len 192 \
  --epochs 10 \
  --batch_size 32 \
  --device cuda \
  --run_name hw_rbt3_crf_hard_e10
```

> **为什么改成 10 epoch**：本子集 1,345 条训练句、batch 32 只有 42 步/epoch，
> 3 epoch = 129 步。实测 3 epoch 会退化成全 `O`（内层验证 span F1 = 0，dev 397 句
> 一句触发词都没抽到），指标全 0，不能写进结果表。10 epoch（430 步）才正常收敛。
> 最优 epoch 由 train 切出的 10% 内层验证集选，dev 不参与选点。
> 如果机器上已有本地权重，把 `--bert_path hfl/rbt3` 换成权重目录即可（脚本不再联网）。

指标文件：`outputs/metrics/disc/hw_rbt3_crf_e10.json`（或 `hw_rbt3_linear_e10.json` / `hw_rbt3_crf_hard_e10.json`）。
读取：

```bash
python -c "import json; d=json.load(open('outputs/metrics/disc/hw_rbt3_crf_e10.json', encoding='utf-8')); print(d['trigger']['f1'], d['argument']['f1'], d['event_exact']['f1'], d['type_macro_f1'])"
```

结果已补进 README 的「完整事件」表和「非法 BIO 统计」表：

| 方法 | Trigger F1 | Argument F1 | Event F1 | Type Macro-F1 |
|---|---:|---:|---:|---:|
| 规则 / 词典 + 正则（L0） | 0.6263 | 0.1413 | 0.0076 | 0.7891 |
| rbt3 + Linear | 0.6931 | 0.3307 | 0.0368 | 0.6136 |
| rbt3 + CRF | 0.7209 | 0.3432 | 0.0337 | 0.6271 |
| rbt3 + CRF + 硬约束 | 0.8070 | 0.4977 | 0.0891 | 0.7142 |

`outputs/checkpoints/*.pt` 约 150 MB/个，**不要提交**（推之前确认 `git status` 里没有 `*.pt`）。

然后：

```bash
cd ../..        # 回到仓库根目录
git status
git add 黄俊杰/week07
git commit -m "黄俊杰-第七周作业：GPU 补充 rbt3+CRF 完整事件结果"
git push fork week07
```

PR #99 会自动更新；不需要新建 PR。
