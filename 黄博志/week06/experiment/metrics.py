"""
统一的评测口径：所有方法都用这里的 evaluate()，避免各脚本各算各的。

主指标 Macro-F1（15 类等权），辅指标 Accuracy / Weighted-F1。
另外保存每类 P/R/F1、混淆矩阵、ECE（有概率时），供报告和错误分析使用。
"""

import json
from math import comb

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def ece_score(y_prob, y_true, n_bins: int = 10):
    """Expected Calibration Error：按最大置信度分箱，|acc - conf| 加权平均。"""
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate(y_true, y_pred, y_prob=None, labels=None):
    """返回可直接写 metrics.json 的 dict。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if labels is None:
        labels = list(range(int(max(y_true.max(), y_pred.max())) + 1))
    labels = list(labels)

    n = int(len(y_true))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        str(lab): {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i, lab in enumerate(labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()

    out = {
        "n": n,
        "acc": round(acc, 6),
        # 本实验所有方法都会输出一个标签，没有解析失败/弃权；
        # 保留这个字段是为了和课程原 summary 的“准确率(含失败)”口径对齐。
        "acc_incl_failed": round(acc, 6),
        "n_failed": 0,
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        # 测试集上实际预测到的不同类别数，用来发现模型塌缩到少数类。
        "coverage": len(set(int(x) for x in y_pred.tolist())),
        "coverage_str": f"{len(set(int(x) for x in y_pred.tolist()))}/{len(labels)}",
        "per_class": per_class,
        "confusion_matrix": cm,
        "ece": None,
        "nll": None,
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if y_prob.ndim == 2 and y_prob.shape[0] == n:
            # 归一化，防止某些模型输出未归一化分数
            row_sum = y_prob.sum(axis=1, keepdims=True)
            y_prob = np.divide(y_prob, row_sum, out=np.full_like(y_prob, 1.0 / y_prob.shape[1]), where=row_sum > 0)
            out["ece"] = round(ece_score(y_prob, y_true), 6)
            out["nll"] = round(float(log_loss(y_true, y_prob, labels=labels)), 6)

    return out


def mcnemar_exact(y_true, pred_a, pred_b):
    """
    配对 McNemar 检验（精确二项，双侧）。
    返回 (b, c, p_value)：
      b = A 对 B 错；c = A 错 B 对。
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return b, c, 1.0
    # 双侧精确二项：2 * P(X <= min(b,c)), X~Bin(n, 0.5)
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return b, c, float(min(1.0, 2.0 * tail))


def md_table(headers, rows):
    """生成 GitHub Markdown 表格；浮点数保留 4 位。"""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def top_confusions(cm, labels, k: int = 10):
    """返回非对角线的 Top-k 混淆对：真实类 -> 预测类, 次数。"""
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i][j] > 0:
                pairs.append((cm[i][j], labels[i], labels[j]))
    pairs.sort(reverse=True)
    return pairs[:k]


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
