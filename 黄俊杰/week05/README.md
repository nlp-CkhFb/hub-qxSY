# week05 作业：基于语言模型的 SFT 训练

- 任务：在 week04 的字符级 Transformer（1.30 M）上做 SFT，数据 `sft_train.jsonl` / `sft_val.jsonl`（3000 / 200 条）
- 模板：`指令：{instruction}\n回答：{output}<eos>`；只对回答部分算 loss，指令部分 labels 置 `-100`
- 模型：沿用 week04 权重，词表 4000 → 4108（补 SFT 新字和 `<eos>`），位置编码 64 → 160，参数量 1.34 M
- 训练：Adam，lr 5e-4，batch 32，10 轮；验证 loss 3.86 → 3.21，回答 token 准确率 28.1% → 39.0%
- SFT 前（基座只会续写）：

```
指令：计算一组数字的平均值和标准差。 数字：23, 45, 67, 89, 12
生成：14年199995年1997年，“国家庭教育”，他们将在100年来，以上海南京东西藏”……
```

- SFT 后（按回答格式生成，遇到 `<eos>` 停止）：

```
指令：计算一组数字的平均值和标准差。 数字：23, 45, 67, 89, 12
生成：这组数字的平均值为 8.8。

指令：提供一个数字列表，返回其中最大和最小数的差值。 数字列表: [3, 7, 14, 2, 9]
生成：最大值为28。
```

- 说明：1.3 M 字符级模型容量有限，上面回答里的数值并不正确；本作业重点验证 SFT 的模板、loss mask 和停止行为。
- 运行：`python sft.py`；单条生成 `python sft.py --load sft_model.pt --prompt "你的指令"`
- 文件：`sft.py`、`language_model.py`、`sft_train.jsonl`、`sft_val.jsonl`、`sft_model.pt`、`sft-curve.png`
