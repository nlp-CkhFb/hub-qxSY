# 任务 1 的 LLM 分支：启动本地 llama.cpp 服务并跑 zero-shot 抽取
#
# 作用：
#   用刚下载的 Ternary-Bonsai-2-27B（Q2_0 GGUF）作为 L3 基线，跑 dev 的 20 条句子，
#   结果与规则 / BERT / 两段式共用 common/evaluate_events.py 的评测口径。
#
# 为什么用 Vulkan 版：
#   llama.cpp 的 CUDA 构建在 Q2_0 量化内核上会卡死（实测），Vulkan 版正常，
#   RTX 4060 Laptop 8GB 下 -ngl 44 约 4 tok/s。
#
# 为什么关思考：
#   Bonsai 是推理模型，默认先输出 thinking；不关的话 256 token 会被思考吃光，正文为空。
#   llama-server 认的是 chat_template_kwargs，所以通过 LLM_CHAT_TEMPLATE_KWARGS 传入。
#
# 运行方式（在 homework 目录下，需要先空出显卡）：
#   powershell -ExecutionPolicy Bypass -File tasks\run_llm_local.ps1
#   powershell ... tasks\run_llm_local.ps1 -N 10 -Tag bonsai_zero_n10   # 想跑更快就减样本

param(
    [int]$N = 20,
    [string]$Tag = "bonsai_zero_n20",
    [int]$MaxTokens = 256
)

$ErrorActionPreference = "Stop"
$PY = "F:\Miniconda3\envs\hf\python.exe"
$EXE = "F:\llama.cpp\b12000-vulkan\llama-server.exe"
$GGUF = "F:\huggingface_cache\models\Ternary-Bonsai-2-27B-gguf\Ternary-Bonsai-2-27B-Q2_0.gguf"
$LOG = "C:\Users\CK\.cache\trae_models\llama_server.log"

$srv = Start-Process -FilePath $EXE `
    -ArgumentList @("-m", $GGUF, "-ngl", "44", "-c", "4096",
                    "--host", "127.0.0.1", "--port", "8080", "--alias", "bonsai", "--jinja") `
    -PassThru -WindowStyle Hidden -RedirectStandardOutput $LOG
Write-Host "等待 llama-server 就绪（加载 7.6GB 模型）..."
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 3
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -TimeoutSec 2
        if ($h.status -eq "ok") { $ready = $true; break }
    } catch {}
}
if (-not $ready) { Stop-Process -Id $srv.Id -Force; throw "llama-server 启动失败，见 $LOG" }

$env:LLM_API_KEY = "local"
$env:LLM_BASE_URL = "http://127.0.0.1:8080/v1"
$env:LLM_MODEL = "bonsai"
$env:LLM_DISABLE_THINKING = "0"
$env:LLM_CHAT_TEMPLATE_KWARGS = '{"enable_thinking": false}'
# 本机回环端口会被 Trae 的 SDK 代理拦成 502，改用 requests 直连
$env:LLM_HTTP_DIRECT = "1"

try {
    & $PY llm_zero_shot\llm_extract.py --backend api --mode zero --n $N `
        --max_tokens $MaxTokens --tag $Tag
} finally {
    Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
}
