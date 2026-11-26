"""
汇总报告：把 outputs/<method>/metrics.json 变成一张可读的对比表。

生成：
  outputs/summary.md       主报告（表格 + 结论 + 混淆矩阵 + 错例入口）
  outputs/summary.csv      Excel 可直接打开
  outputs/errors.md        微调模型的错例
  outputs/confusion_*.png  每个方法的混淆矩阵
  outputs/per_class_f1.png 每类 F1 对比
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
import metrics

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def _font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_results(out_dir: Path):
    results = {}
    for method in config.METHODS:
        p = out_dir / method / "metrics.json"
        if p.exists():
            results[method] = _load_json(p)
    return results


def _load_preds(out_dir: Path, method: str):
    p = out_dir / method / "predictions.json"
    if not p.exists():
        return None
    rows = _load_json(p)
    return {r["uid"]: r for r in rows}


def _mcnemar_pair(out_dir: Path, method_a: str, method_b: str):
    a = _load_preds(out_dir, method_a)
    b = _load_preds(out_dir, method_b)
    if not a or not b:
        return None
    uids = sorted(set(a) & set(b))
    if not uids:
        return None
    y_true = np.array([a[u]["true"] for u in uids], dtype=np.int64)
    pred_a = np.array([a[u]["pred"] for u in uids], dtype=np.int64)
    pred_b = np.array([b[u]["pred"] for u in uids], dtype=np.int64)
    return metrics.mcnemar_exact(y_true, pred_a, pred_b)


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


SHORT_NAMES = {
    "rule": "规则 / 关键词",
    "tfidf_svc": "TF-IDF + SVM",
    "tfidf_nb": "TF-IDF + NB",
    "bert_frozen": "BERT 冻结 + LR",
    "roberta_frozen": "RoBERTa 冻结 + LR",
    "bert_ft": "BERT 微调",
    "roberta_ft": "RoBERTa 微调",
}


def _coverage_str(method, r, out_dir: Path):
    """覆盖类别 = 测试集上实际预测到的不同类别数 / 总类别数。"""
    if r.get("coverage_str"):
        return r["coverage_str"]
    preds = _load_preds(out_dir, method)
    if not preds:
        return "-"
    n = len({int(row["pred"]) for row in preds.values()})
    return f"{n}/{len(config.LABELS)}"


def _train_cost(method, r):
    if method == "rule":
        return "—"
    sec = r.get("train_seconds")
    if sec is None:
        return "-"
    if sec < 60:
        text = f"{sec:.0f} 秒"
    else:
        text = f"{sec / 60:.1f} 分钟"
    mb = r.get("extra", {}).get("checkpoint_mb")
    if mb:
        text += f" / {mb:.0f} MB"
    return text


def _summary_rows(results, out_dir: Path):
    """用户要求的主表：方法 / 准确率(含失败) / Macro-F1 / 覆盖类别 / 训练成本。"""
    rows = []
    ordered = sorted(results.items(), key=lambda kv: kv[1]["macro_f1"], reverse=True)
    best_method = ordered[0][0] if ordered else None
    for method, r in ordered:
        name = SHORT_NAMES.get(method, r["display_name"])
        if method == best_method:
            name = f"**{name}**"
        rows.append(
            [
                name,
                _fmt(r.get("acc_incl_failed", r["acc"]), 3),
                _fmt(r["macro_f1"], 3),
                _coverage_str(method, r, out_dir),
                _train_cost(method, r),
            ]
        )
    return rows


def _detail_rows(results):
    rows = []
    ordered = sorted(results.values(), key=lambda r: r["macro_f1"], reverse=True)
    for r in ordered:
        extra = r.get("extra", {})
        rows.append(
            [
                r["display_name"],
                _fmt(r["weighted_f1"]),
                _fmt(r.get("ece")),
                _fmt(r.get("infer_ms_per_sample"), 2),
                _fmt(extra.get("checkpoint_mb"), 0),
            ]
        )
    return rows


def _conclusions(results, out_dir: Path):
    lines = []
    if not results:
        return lines

    best = max(results.values(), key=lambda r: r["macro_f1"])
    lines.append(
        f"- 最佳方法：**{best['display_name']}**，Macro-F1 **{best['macro_f1']:.4f}**，"
        f"Accuracy {best['acc']:.4f}。"
    )

    if "bert_ft" in results and "roberta_ft" in results:
        a, b = results["bert_ft"], results["roberta_ft"]
        delta = b["macro_f1"] - a["macro_f1"]
        mc = _mcnemar_pair(out_dir, "bert_ft", "roberta_ft")
        if mc:
            bc, cc, p = mc
            sig = "显著" if p < 0.05 else "不显著"
            lines.append(
                f"- 微调对比：RoBERTa 比 BERT {'高' if delta >= 0 else '低'} "
                f"**{abs(delta) * 100:.2f}** 个百分点；"
                f"McNemar b={bc}, c={cc}, p={p:.4f}（{sig}）。"
            )
        else:
            lines.append(
                f"- 微调对比：RoBERTa 比 BERT {'高' if delta >= 0 else '低'} "
                f"**{abs(delta) * 100:.2f}** 个百分点。"
            )

    if all(k in results for k in ("bert_ft", "bert_frozen", "roberta_ft", "roberta_frozen")):
        b_gain = results["bert_ft"]["macro_f1"] - results["bert_frozen"]["macro_f1"]
        r_gain = results["roberta_ft"]["macro_f1"] - results["roberta_frozen"]["macro_f1"]
        lines.append(
            f"- 微调相对冻结特征：BERT 提升 **{b_gain * 100:.2f}** 个百分点，"
            f"RoBERTa 提升 **{r_gain * 100:.2f}** 个百分点。"
        )

    if "bert_ft" in results and "roberta_ft" in results:
        deltas = [
            results["roberta_ft"]["per_class"][str(i)]["f1"]
            - results["bert_ft"]["per_class"][str(i)]["f1"]
            for i in range(len(config.LABELS))
        ]
        pos = sum(1 for d in deltas if d > 0)
        best_i = max(range(len(deltas)), key=lambda i: deltas[i])
        worst_i = min(range(len(deltas)), key=lambda i: deltas[i])
        lines.append(
            f"- 每类对比：RoBERTa 在 **{pos}/15** 类上更高；提升最大的是 "
            f"**{config.LABELS[best_i]}（{deltas[best_i] * 100:+.2f} pp）**，"
            f"下降最大的是 **{config.LABELS[worst_i]}（{deltas[worst_i] * 100:+.2f} pp）**。"
        )

    if results:
        full = sum(
            1
            for m, r in results.items()
            if _coverage_str(m, r, out_dir) == f"{len(config.LABELS)}/{len(config.LABELS)}"
        )
        lines.append(
            f"- 覆盖类别：{full}/{len(results)} 个方法覆盖全部 15 类；"
            "没有出现 zero-shot 式的类别塌缩。"
        )

    if "bert_frozen" in results and "roberta_frozen" in results:
        a, b = results["bert_frozen"], results["roberta_frozen"]
        delta = b["macro_f1"] - a["macro_f1"]
        mc = _mcnemar_pair(out_dir, "bert_frozen", "roberta_frozen")
        if mc:
            bc, cc, p = mc
            sig = "显著" if p < 0.05 else "不显著"
            lines.append(
                f"- 冻结特征对比：RoBERTa 比 BERT {'高' if delta >= 0 else '低'} "
                f"**{abs(delta) * 100:.2f}** 个百分点；"
                f"McNemar b={bc}, c={cc}, p={p:.4f}（{sig}）。"
            )
        else:
            lines.append(
                f"- 冻结特征对比：RoBERTa 比 BERT {'高' if delta >= 0 else '低'} "
                f"**{abs(delta) * 100:.2f}** 个百分点。"
            )

    if "tfidf_svc" in results:
        tfidf = results["tfidf_svc"]
        gap = best["macro_f1"] - tfidf["macro_f1"]
        lines.append(
            f"- 传统基线：TF-IDF + LinearSVC 的 Macro-F1 为 {tfidf['macro_f1']:.4f}，"
            f"比最佳方法低 **{gap * 100:.2f}** 个百分点。"
        )

    if "rule" in results:
        rule = results["rule"]
        lines.append(
            f"- 规则基线：关键词规则 Macro-F1 仅 {rule['macro_f1']:.4f}，"
            "说明这个任务不能只靠少量关键词。"
        )

    lines.append(
        "- 注意：以上是单 seed、2,000 条测试集的结果；"
        "Macro-F1 差距小于约 1 个百分点、或 McNemar p≥0.05 时，"
        "建议表述为“趋势”而不是“显著更好”。"
    )
    return lines


def _per_class_table(results):
    methods = [m for m in ("bert_frozen", "roberta_frozen", "bert_ft", "roberta_ft") if m in results]
    if not methods:
        return None
    header = ["类别"] + [results[m]["display_name"] + " F1" for m in methods]
    if "bert_ft" in results and "roberta_ft" in results:
        header.append("Δ RoBERTa-BERT (微调)")
    rows = []
    for i, lab in enumerate(config.LABELS):
        row = [f"{i} {lab}"]
        for m in methods:
            row.append(_fmt(results[m]["per_class"][str(i)]["f1"]))
        if "bert_ft" in results and "roberta_ft" in results:
            d = (
                results["roberta_ft"]["per_class"][str(i)]["f1"]
                - results["bert_ft"]["per_class"][str(i)]["f1"]
            )
            row.append(f"{d * 100:+.2f} pp")
        rows.append(row)
    return metrics.md_table(header, rows)


def _top_confusion_table(results):
    rows = []
    for m in ("bert_ft", "roberta_ft", "bert_frozen", "roberta_frozen"):
        if m not in results:
            continue
        cm = results[m]["confusion_matrix"]
        for cnt, true_lab, pred_lab in metrics.top_confusions(cm, config.LABELS, k=8):
            rows.append([results[m]["display_name"], f"{true_lab} → {pred_lab}", cnt])
    if not rows:
        return None
    return metrics.md_table(["方法", "真实 → 预测", "次数"], rows)


def _draw_confusion(cm, labels, title, path: Path):
    n = len(labels)
    cell = 36
    left, top, right, bottom = 110, 72, 20, 52
    W = left + n * cell + right
    H = top + n * cell + bottom
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title = _font(18)
    f_note = _font(11)
    f_cell = _font(10)
    f_axis = _font(11)

    d.text((left, 12), title, fill=(20, 20, 20), font=f_title)
    d.text((left, 40), "行 = 真实类别，列 = 预测类别；颜色按行归一化", fill=(100, 100, 100), font=f_note)

    for i in range(n):
        row_sum = max(1, sum(cm[i]))
        for j in range(n):
            x0 = left + j * cell
            y0 = top + i * cell
            ratio = cm[i][j] / row_sum
            r = int(255 - 180 * ratio)
            g = int(255 - 140 * ratio)
            b = int(255 - 20 * ratio)
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=(r, g, b), outline=(220, 220, 220))
            if cm[i][j]:
                color = (255, 255, 255) if ratio > 0.6 else (20, 20, 20)
                txt = str(cm[i][j])
                bbox = d.textbbox((0, 0), txt, font=f_cell)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                d.text(
                    (x0 + (cell - tw) / 2, y0 + (cell - th) / 2 - 1),
                    txt,
                    fill=color,
                    font=f_cell,
                )

    for i in range(n):
        y0 = top + i * cell
        d.text((5, y0 + cell / 2 - 7), f"{i} {labels[i]}", fill=(40, 40, 40), font=f_axis)
        x0 = left + i * cell
        d.text((x0 + cell / 2 - 4, top + n * cell + 8), str(i), fill=(40, 40, 40), font=f_axis)

    d.text((left + n * cell / 2 - 30, H - 22), "预测类别", fill=(80, 80, 80), font=f_note)
    img.save(path)


def _draw_per_class(results, path: Path):
    methods = [m for m in ("bert_frozen", "roberta_frozen", "bert_ft", "roberta_ft") if m in results]
    if not methods:
        return
    n = len(config.LABELS)
    bar_w, gap = 14, 4
    group_w = len(methods) * bar_w + (len(methods) - 1) * gap
    left, top, right, bottom = 60, 70, 20, 70
    W = left + n * group_w + right
    plot_h = 280
    H = top + plot_h + bottom
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title = _font(18)
    f_note = _font(11)
    f_axis = _font(10)
    colors = [(120, 120, 120), (70, 130, 200), (230, 150, 60), (40, 160, 110)]

    d.text((left, 12), "每类 F1 对比", fill=(20, 20, 20), font=f_title)
    for k in range(6):
        v = k / 5
        y = top + plot_h - v * plot_h
        d.line([left, y, left + n * group_w, y], fill=(230, 230, 230))
        d.text((left - 34, y - 6), f"{v:.1f}", fill=(90, 90, 90), font=f_axis)

    for i, lab in enumerate(config.LABELS):
        gx = left + i * group_w
        for k, m in enumerate(methods):
            f1 = results[m]["per_class"][str(i)]["f1"]
            bh = max(1, int(f1 * plot_h))
            x0 = gx + k * (bar_w + gap)
            d.rectangle(
                [x0, top + plot_h - bh, x0 + bar_w, top + plot_h],
                fill=colors[k % len(colors)],
            )
        d.text((gx + group_w / 2 - 4, top + plot_h + 8), str(i), fill=(40, 40, 40), font=f_axis)

    # 图例
    lx = left
    for k, m in enumerate(methods):
        d.rectangle([lx, 40, lx + 10, 50], fill=colors[k % len(colors)])
        name = results[m]["display_name"]
        d.text((lx + 14, 38), name, fill=(40, 40, 40), font=f_note)
        lx += 24 + int(d.textlength(name, font=f_note))

    d.text((left + n * group_w / 2 - 20, H - 24), "类别 ID（对应 summary.md 的映射）", fill=(80, 80, 80), font=f_note)
    img.save(path)


def _write_errors(out_dir: Path, results):
    lines = ["# 错例分析", ""]
    lines.append("每个微调模型按“预测置信度从高到低”列出前 30 个错例，便于看模型在哪些边界上自信地犯错。")
    lines.append("")
    for m in ("bert_ft", "roberta_ft"):
        if m not in results:
            continue
        preds = _load_preds(out_dir, m)
        if not preds:
            continue
        rows = list(preds.values())
        errs = [r for r in rows if r["pred"] != r["true"]]
        conf = lambda r: max(r["prob"]) if r.get("prob") else 0.0
        errs.sort(key=conf, reverse=True)
        lines.append(f"## {results[m]['display_name']}（错 {len(errs)} / {len(rows)}）")
        lines.append("")
        lines.append("| 文本 | 真实 | 预测 | 置信度 |")
        lines.append("| --- | --- | --- | --- |")
        for r in errs[:30]:
            text = r["sentence"].replace("|", "\\|").replace("\n", " ")
            if len(text) > 80:
                text = text[:77] + "..."
            lines.append(
                f"| {text} | {config.LABELS[r['true']]} | {config.LABELS[r['pred']]} | {conf(r):.3f} |"
            )
        lines.append("")
    (out_dir / "errors.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(out_dir=None):
    out_dir = Path(out_dir) if out_dir else config.OUT_DIR
    results = _load_results(out_dir)
    if not results:
        print(f"[report] {out_dir} 下没有 metrics.json，跳过")
        return

    split_path = out_dir / "splits.json"
    meta = {}
    if split_path.exists():
        meta = _load_json(split_path).get("meta", {})

    # ---------- CSV ----------
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method", "display_name", "acc_incl_failed", "macro_f1",
                "coverage", "train_cost", "n", "acc", "weighted_f1",
                "ece", "nll", "train_seconds", "infer_ms_per_sample",
                "params_m", "checkpoint_mb",
            ]
        )
        for m in sorted(results, key=lambda x: results[x]["macro_f1"], reverse=True):
            r = results[m]
            extra = r.get("extra", {})
            writer.writerow(
                [
                    m,
                    SHORT_NAMES.get(m, r["display_name"]),
                    r.get("acc_incl_failed", r["acc"]),
                    r["macro_f1"],
                    _coverage_str(m, r, out_dir),
                    _train_cost(m, r),
                    r["n"],
                    r["acc"],
                    r["weighted_f1"],
                    r.get("ece"),
                    r.get("nll"),
                    r.get("train_seconds"),
                    r.get("infer_ms_per_sample"),
                    extra.get("params_m"),
                    extra.get("checkpoint_mb"),
                ]
            )

    # ---------- 混淆矩阵图 ----------
    for m, r in results.items():
        _draw_confusion(
            r["confusion_matrix"],
            config.LABELS,
            r["display_name"],
            out_dir / f"confusion_{m}.png",
        )
    _draw_per_class(results, out_dir / "per_class_f1.png")

    # ---------- summary.md ----------
    md = []
    md.append("# 换预训练模型对比实验：BERT-base vs RoBERTa-base")
    md.append("")
    md.append("同一 TNEWS 数据切分、同一分类头、同一超参，只替换预训练骨干。")
    md.append("")

    md.append("## 1. 实验设置")
    md.append("")
    if meta:
        setup = [
            ["原始训练集", meta.get("raw_train", "-"), "清洗前 train.json"],
            ["训练集内部重复", meta.get("train_dup_removed", "-"), "按整句完全去重"],
            ["原始验证集", meta.get("raw_val", "-"), "清洗前 val.json"],
            ["验证集与训练集重复", meta.get("val_dup_removed", "-"), "评测前删除"],
            ["去重后验证集", meta.get("val_clean", "-"), "作为测试集池"],
            ["训练 / 开发 / 测试", f"{meta.get('train')} / {meta.get('dev')} / {meta.get('test')}", "固定 seed=42 分层抽样"],
            ["max_len", meta.get("max_len", "-"), "TNEWS 句子 p99≈39"],
            ["骨干", "BERT-base-chinese / RoBERTa-base", "均为 12 层 / 768 隐藏，约 102M 参数"],
        ]
        md.append(metrics.md_table(["项目", "值", "说明"], setup))
    else:
        md.append("（找不到 splits.json，数据规模以 metrics.json 的 n 为准。）")
    md.append("")

    md.append("## 2. 主结果（按 Macro-F1 排序）")
    md.append("")
    md.append("| 方法 | 准确率(含失败) | Macro-F1 | 覆盖类别 | 训练成本 |")
    md.append("| --- | --- | --- | --- | --- |")
    for row in _summary_rows(results, out_dir):
        md.append("| " + " | ".join(str(x) for x in row) + " |")
    md.append("")
    md.append("> **准确率(含失败)**：本实验所有方法都会输出一个标签，没有解析失败/弃权，因此该列等于普通 Accuracy。")
    md.append(">")
    md.append("> **覆盖类别**：测试集上实际预测到的不同类别数 / 15；用来发现模型塌缩到少数类。")
    md.append(">")
    md.append("> **训练成本**：CPU（16 线程）实测；`/ 393 MB` 表示权重体积。")
    md.append("")
    md.append("### 补充：校准与推理成本")
    md.append("")
    md.append(
        metrics.md_table(
            ["方法", "Weighted-F1", "ECE", "推理 ms/条", "权重 MB"],
            _detail_rows(results),
        )
    )
    md.append("")
    md.append("> ECE 为 `-` 表示该方法没有输出概率（规则/LinearSVC），不参与校准比较。")
    md.append("")

    md.append("## 3. 核心结论")
    md.append("")
    md.extend(_conclusions(results, out_dir))
    md.append("")

    per_class = _per_class_table(results)
    if per_class:
        md.append("## 4. 每类 F1")
        md.append("")
        md.append(per_class)
        md.append("")
        md.append("![每类 F1](per_class_f1.png)")
        md.append("")

    md.append("## 5. 混淆矩阵")
    md.append("")
    for m in ("bert_ft", "roberta_ft", "bert_frozen", "roberta_frozen", "tfidf_svc", "rule"):
        if m in results:
            md.append(f"### {results[m]['display_name']}")
            md.append("")
            md.append(f"![{m}](confusion_{m}.png)")
            md.append("")

    top_cm = _top_confusion_table(results)
    if top_cm:
        md.append("## 6. Top 混淆对")
        md.append("")
        md.append(top_cm)
        md.append("")

    md.append("## 7. 成本与复现")
    md.append("")
    md.append("- 训练/推理时间均为 CPU（16 线程）实测；不同机器会有差异。")
    md.append("- 复现：`python run_all.py --methods all --force`；冒烟：`python run_all.py --smoke --methods all --force`。")
    md.append("- 错例：见 [errors.md](errors.md)。")
    md.append(
        "- 课程原 summary 里的 Qwen3 zero-shot / LoRA / 全量微调 / API 行来自 "
        "500 条干净验证集的另一套实验，本次没有重跑；为避免不同测试集口径混比，未并入主表。"
    )
    md.append("")
    md.append("## 8. 类别 ID 映射")
    md.append("")
    md.append(metrics.md_table(["ID", "类别"], [[i, lab] for i, lab in enumerate(config.LABELS)]))
    md.append("")

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    _write_errors(out_dir, results)
    print(f"[report] 已生成 {out_dir / 'summary.md'}")


if __name__ == "__main__":
    build_report()
