"""
换预训练模型对比实验：全局配置

实验目标：
  在完全相同的 TNEWS 数据切分、预处理、训练超参和评测口径下，
  比较 bert-base-chinese 与 nlp_roberta_backbone_base_std 两个同规模
  （12 层 / 768 隐藏）中文预训练骨干，回答：
    1. 只换预训练模型，微调后的 Macro-F1 能差多少？
    2. 不微调、只把 [CLS] 当特征，两个骨干的差距又是多少？
    3. 换模型带来的收益，值不值它多花的时间/体积？

所有路径都可以用环境变量覆盖，方便换机器运行。
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- 目录
EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent                 # ...\week06
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FEATURE_DIR = OUT_DIR / "features"

# 课程原始 TNEWS 数据（如果 week06/data 下没有，就从这里读）
COURSE_DATA_DIR = Path(
    os.getenv(
        "TNEWS_DATA_DIR",
        r"D:\八斗\week6\第六周 分类任务\week6-数据接入与分流\code\data",
    )
)

# ---------------------------------------------------------------- 模型
# 原课程使用的 BERT-base-chinese
BERT_DIR = Path(
    os.getenv(
        "BERT_DIR",
        r"D:\八斗\week4\第四周 语言模型\bert-base-chinese",
    )
)

# 本次要换上的 RoBERTa-base（ModelScope，旧版 LayerNorm 命名，加载时需转换）
ROBERTA_DIR = Path(
    os.getenv(
        "ROBERTA_DIR",
        str(ROOT / "nlp_roberta_backbone_base_std"),
    )
)

# ---------------------------------------------------------------- 数据
# TNEWS 原始 15 类，顺序与官方 label_map 的 name2id 一致
LABELS = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]
NUM_LABELS = len(LABELS)

SEED = 42
TRAIN_SIZE = 7000        # 从训练集分层抽 7k 做训练
DEV_SIZE = 1000          # 从训练集分层抽 1k 做 epoch 选择
TEST_SIZE = 2000         # 从去重后的验证集分层抽 2k 做最终评测
MAX_LEN = 40             # TNEWS 句子 p99≈39，40 足够，且比 64 快很多
BATCH_SIZE = 32
EPOCHS = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0
NUM_THREADS = 16

# 固定方法名，报告和缓存都靠它
METHODS = {
    "rule": "关键词规则",
    "tfidf_svc": "TF-IDF + LinearSVC",
    "tfidf_nb": "TF-IDF + MultinomialNB",
    "bert_frozen": "BERT-base 冻结 [CLS] + LR",
    "roberta_frozen": "RoBERTa-base 冻结 [CLS] + LR",
    "bert_ft": "BERT-base 微调",
    "roberta_ft": "RoBERTa-base 微调",
}

# 两个骨干的显示名与权重路径
BACKBONES = {
    "bert": {"name": "BERT-base-chinese", "dir": BERT_DIR, "size_mb": 393},
    "roberta": {"name": "RoBERTa-base (ModelScope)", "dir": ROBERTA_DIR, "size_mb": 411},
}


def ensure_dirs():
    for p in (OUT_DIR, FEATURE_DIR):
        p.mkdir(parents=True, exist_ok=True)
