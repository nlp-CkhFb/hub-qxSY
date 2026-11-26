# Week7 信息抽取 · 课后作业

本目录是本次作业的**完整提交包**：代码、数据副本、实跑产物都在这里，打包本文件夹即可提交。

## 1. 环境与模型

| 项目 | 路径 / 说明 |
|---|---|
| Python | `F:\Miniconda3\envs\hf\python.exe`（torch 2.14.1+cu126） |
| 判别式预训练模型 | `F:\huggingface_cache\models\roberta`（hfl/chinese-roberta-wwm-ext，中文 RoBERTa；config 沿用 BERT 架构，权重经 sha256 校验为官方 RoBERTa，而非 bert-base-chinese） |
| 生成式模型 | `F:\huggingface_cache\models\Ternary-Bonsai-2-27B-gguf\Ternary-Bonsai-2-27B-Q2_0.gguf`（27B 三值量化，7.6GB） |
| 推理引擎 | `F:\llama.cpp\b12000-vulkan\llama-server.exe`（Q2_0 内核用 CUDA 版会挂起，必须用 Vulkan 版） |
| 数据 | `data/processed/`（train 5747 句 / dev 930 句 / dev_docs 985 篇） |

## 2. 目录结构

```
homework/
├─ README.md                 本文件
├─ run_all.ps1               一键复现四路方法 + 四个分析任务
├─ data/                     数据副本（processed / _mock / sample）
├─ common/ bert/ bert_pipeline/ rule/ llm_zero_shot/ llm_lora/ service/ eval/ tools/
│                            课程原始代码（未改动，指标口径统一走 common/evaluate_events.py）
├─ tasks/                    本次作业新写的 4 个脚本（每个文件头都有用途与运行方式）
│   ├─ task1_compare.py      四路方法 → 同一张对比表
│   ├─ task2_badcase.py      挑 10 条 badcase 并按三类归档
│   ├─ task3_event_stream.py extract_event() 接入事件流 demo（一条公告 → 事件列表）
│   └─ task4_loss_curve.py   读 steps.jsonl 画 loss 曲线
└─ outputs/
    ├─ metrics/summary_4methods.md / .csv   任务 1 产物
    ├─ badcase/badcase_10.md                任务 2 产物
    ├─ demo/event_stream_demo.json          任务 3 产物
    ├─ figures/loss_curve.png / .csv        任务 4 产物
    ├─ metrics/disc|llm/*.json              各方法原始指标
    ├─ disc/pred_*.jsonl                    各方法 dev 预测
    └─ logs/*_steps.jsonl / *_train.json    训练曲线与日志
```

## 3. 四路方法怎么跑（都在 homework 目录下执行）

```powershell
$PY = "F:\Miniconda3\envs\hf\python.exe"
$BERT = "F:\huggingface_cache\models\roberta"

# ① L0 规则：词典 + 正则（纯 CPU，秒级）
& $PY rule\run_rule.py

# ② L2 BERT + Linear（联合 149 标签；本次 3 epoch / batch 16）
& $PY bert\train.py --decoder linear --bert_path $BERT --epochs 3 --batch_size 16 `
    --run_name bert_linear_256 --log_every 50

# ③ L2 两段式：检测器 → 类型分类 → 13 个类型条件论元专家
& $PY bert_pipeline\train.py --bert_path $BERT

# ④ L3 本地 LLM zero-shot（llama-server 提供 OpenAI 兼容接口）
& "F:\llama.cpp\b12000-vulkan\llama-server.exe" -m "<GGUF 路径>" -ngl 44 -c 3072 `
    --host 127.0.0.1 --port 8080 --alias bonsai
$env:LLM_API_KEY="local"; $env:LLM_BASE_URL="http://127.0.0.1:8080/v1"
$env:LLM_MODEL="bonsai"; $env:LLM_DISABLE_THINKING="0"
& $PY llm_zero_shot\llm_extract.py --backend api --mode zero --n 20 --max_tokens 256 --tag bonsai_zero_n20
```

## 4. 四个分析任务怎么跑

```powershell
& $PY tasks\task1_compare.py --llm_tag bonsai_zero_n20   # → outputs/metrics/summary_4methods.md
& $PY tasks\task2_badcase.py --source bert               # → outputs/badcase/badcase_10.md
& $PY tasks\task3_event_stream.py                        # → outputs/demo/event_stream_demo.json
& $PY tasks\task4_loss_curve.py --run bert_linear_256    # → outputs/figures/loss_curve.png
```

也可以直接 `powershell -ExecutionPolicy Bypass -File run_all.ps1` 按顺序全跑一遍。

## 5. 结果速览

统一评测口径：`common/evaluate_events.py`，dev_extract 930 句（LLM 为 dev 采样的 20 条，n=20）。

| 方法（层级） | trigger F1 | argument F1 | event F1 | 训练/成本 | 延迟 |
|---|---|---|---|---|---|
| 规则：词典 + 正则（L0） | 0.6168 | 0.1740 | 0.0161 | 0（纯 CPU） | 0.74 ms |
| BERT + Linear（L2，联合 149 标签） | **0.8766** | **0.6431** | **0.2413** | 587 s（12 层 RoBERTa / 3 epoch） | 9.23 ms |
| 两段式 检测→类型→论元（L2） | 0.7375 | 0.3623 | 0.0843 | 1571 s（16 个模型） | 19.83 ms |
| 本地 LLM zero-shot（L3，Bonsai-27B，n=20） | 0.6111 | 0.4941 | 0.0000 | 20 条 / 1037 s | 51850 ms |

> 两段式本次为 8GB 显存适配版（检测 batch 16 / max_len 128），略低于 PPT 基线（0.677/0.369/0.096）；
> LLM 为 20 条子集，event_exact 恰好为 0，参考时看 trigger/argument 两列。

四类产物：

- 任务 1 → [summary_4methods.md](outputs/metrics/summary_4methods.md)（表）+ `.csv`
- 任务 2 → [badcase_10.md](outputs/badcase/badcase_10.md)（边界 4 / 角色串味 4 / 触发词歧义 2）
- 任务 3 → [event_stream_demo.json](outputs/demo/event_stream_demo.json)（7 句 → 1 个「被约谈」事件，含 3 个论元）
- 任务 4 → [loss_curve.png](outputs/figures/loss_curve.png)（loss 5.13 → 0.17，3 epoch）

## 6. 环境踩坑记录

1. **torch 与 torchvision/torchaudio 版本不匹配**：环境里 torch 是用户级 2.14.1，而 env 内 torchvision 0.21+cu124 / torchaudio 2.6.0+cu124 与之不兼容，
   `import transformers` 会因 `torchvision::nms` 不存在而失败。本项目只做文本任务，已卸载这两个包。
2. **Q2_0 量化内核**：llama.cpp 的 CUDA 构建跑该 GGUF 会卡死；Vulkan 构建正常（`-ngl 44` 约 4 tok/s），故 LLM 后端统一用 Vulkan 版。
3. **模型路径**：`model_paths.py` 原本在 `pretrain_models/` 下找 BERT，本作业全部显式传 `--bert_path`，避免依赖目录约定。
4. **本地 LLM 的三处坑**：
   - OpenAI SDK 访问本机回环端口会被代理层拦成 502，改用 `requests` 直连（脚本里 `LLM_HTTP_DIRECT=1`）；
   - Bonsai 是推理模型，必须在请求里带 `chat_template_kwargs.enable_thinking=false`，否则 256 个 token 全被思考吃光、正文为空；
   - llama-server 是当前 shell 的子进程，shell 被回收时服务会一起消失，所以 `tasks/run_llm_local.ps1` 把「起服务 → 评测 → 关服务」放在同一次执行里。
5. **8GB 显存下的两段式**：检测器 batch=64 会触发显存颠簸（实测 2 小时跑不完一轮），改为 batch=16 / max_len=128 后 26 分钟跑完 16 个模型。
