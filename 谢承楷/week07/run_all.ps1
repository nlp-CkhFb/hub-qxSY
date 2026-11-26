# 一键复现脚本（Windows PowerShell）
# 作用：在我这台机器上从零跑完四路方法 + 三个分析任务。
# 运行：在 homework 目录下执行  powershell -ExecutionPolicy Bypass -File run_all.ps1
# 注意：① 需要 NVIDIA 显卡（BERT 训练 ~15 分钟，两段式 ~7 分钟）；
#       ② LLM 用 F:\llama.cpp 的 Vulkan 版 llama-server 起本地服务，27B Q2_0 在 8GB 卡上约 4 tok/s。

$ErrorActionPreference = "Stop"
$PY = "F:\Miniconda3\envs\hf\python.exe"
$BERT = "F:\huggingface_cache\models\roberta"

# 1) 规则基线（CPU，秒级）
& $PY rule\run_rule.py

# 2) BERT + Linear（联合 149 标签）
& $PY bert\train.py --decoder linear --bert_path $BERT --epochs 3 --batch_size 16 `
    --run_name bert_linear_256 --log_every 50

# 3) 两段式：检测 → 类型 → 13 个类型条件论元专家
#    8GB 显存下 batch 64 会触发显存颠簸（实测 2 小时都跑不完），故降为 16 / max_len 128
& $PY bert_pipeline\train.py --bert_path $BERT --cls_batch_size 16 --cls_max_len 128 `
    --arg_batch_size 16 --cls_epochs 2 --arg_epochs 3 --log_every 100

# 4) 本地 LLM zero-shot（OpenAI 兼容接口；起服务/关思考/收尾都封装在脚本里）
& powershell -ExecutionPolicy Bypass -File tasks\run_llm_local.ps1 -N 20 -Tag bonsai_zero_n20

# 5) 三个分析任务
& $PY tasks\task1_compare.py --llm_tag bonsai_zero_n20
& $PY tasks\task2_badcase.py --source bert
& $PY tasks\task3_event_stream.py
& $PY tasks\task4_loss_curve.py --run bert_linear_256
