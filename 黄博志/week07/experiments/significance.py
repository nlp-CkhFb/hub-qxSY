"""
显著性检验：配对 bootstrap（F1 差值） + McNemar（句子级事件完全正确）。

为什么不能只比两个 F1 的小数点：
  dev 只有 930 句，论元 tp/fp/fn 都是几千，但句子级方差很大；两个方法差 0.5pp
  很可能只是同一批难句的随机波动。配对检验（同一批句子、同一批重采样下标）
  比独立置信区间更敏感，也更符合「同一评测集上比较两个模型」的设定。

口径：
  - F1 差值：对句子做有放回重采样 5000 次，每次用重采样后的 tp/fp/fn 重算 F1；
    报告 95% percentile CI 与双侧 p（p = 2*min(P(diff<=0), P(diff>=0))）。
  - McNemar：句子级 event_exact 是否正确（该句所有金标事件都完全命中且无多抽），
    对 2x2 不一致对做精确二项检验（scipy 可用时），否则正态近似。
"""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_events import as_event_list, compare_events, load_jsonl, row_key


def load_pred_map(path):
    rows = load_jsonl(str(path))
    return {row_key(r): as_event_list(r) for r in rows}


def sentence_counts(gold_rows, pred_map, type_filter=None):
    """逐句 (trigger, argument, event) 的 tp/fp/fn；type_filter 用于 per-type 切片。"""
    out = []
    for g in gold_rows:
        g_events = as_event_list(g)
        p_events = pred_map.get(row_key(g), [])
        if type_filter is not None:
            g_events = [e for e in g_events if e["type"] == type_filter]
            p_events = [e for e in p_events if e["type"] == type_filter]
        t, a, e = compare_events(g_events, p_events)
        out.append({"trigger": t, "argument": a, "event": e,
                    "n_gold": len(g_events)})
    return out


def f1_from(items, key):
    tp = sum(x[key][0] for x in items)
    fp = sum(x[key][1] for x in items)
    fn = sum(x[key][2] for x in items)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0, (tp, fp, fn)


def paired_bootstrap(counts_a, counts_b, metric="argument", n_boot=5000, seed=42):
    """同一组重采样下标同时作用于两个方法，得到 F1 差值的分布。"""
    assert len(counts_a) == len(counts_b)
    rng = random.Random(seed)
    N = len(counts_a)
    base_a, _ = f1_from(counts_a, metric)
    base_b, _ = f1_from(counts_b, metric)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(N) for _ in range(N)]
        fa, _ = f1_from([counts_a[i] for i in idx], metric)
        fb, _ = f1_from([counts_b[i] for i in idx], metric)
        diffs.append(fb - fa)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]
    p_le = sum(d <= 0 for d in diffs) / n_boot
    p_ge = sum(d >= 0 for d in diffs) / n_boot
    # bootstrap 的 p 有 1/n_boot 的分辨率下限；报 0 会误导成「p 恰好为 0」
    p = max(min(1.0, 2 * min(p_le, p_ge)), 1.0 / n_boot)
    return {
        "metric": metric,
        "f1_a": round(base_a, 5),
        "f1_b": round(base_b, 5),
        "delta_pp": round((base_b - base_a) * 100, 3),
        "ci95_pp": [round(lo * 100, 3), round(hi * 100, 3)],
        "p_bootstrap": round(p, 5),
        "n_boot": n_boot,
    }


def _mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(b, n, 0.5).pvalue)
    except Exception:
        # 正态近似 + 连续性校正
        z = (abs(b - c) - 1) / math.sqrt(n)
        return float(math.erfc(z / math.sqrt(2)))


def mcnemar(counts_a, counts_b):
    """句子级 event_exact 是否正确：a 对 b 错 / a 错 b 对。"""
    b = c = n_ok_a = n_ok_b = 0
    for xa, xb in zip(counts_a, counts_b):
        ok_a = (xa["event"][0] == xa["n_gold"]) and (xa["event"][1] == 0)
        ok_b = (xb["event"][0] == xb["n_gold"]) and (xb["event"][1] == 0)
        n_ok_a += int(ok_a)
        n_ok_b += int(ok_b)
        if ok_a and not ok_b:
            b += 1
        elif ok_b and not ok_a:
            c += 1
    return {
        "n_correct_a": n_ok_a,
        "n_correct_b": n_ok_b,
        "a_only": b,
        "b_only": c,
        "p_mcnemar": round(_mcnemar_p(b, c), 5),
    }


def compare_pair(name_a, path_a, name_b, path_b, gold_rows, n_boot=5000):
    a_map, b_map = load_pred_map(path_a), load_pred_map(path_b)
    ca = sentence_counts(gold_rows, a_map)
    cb = sentence_counts(gold_rows, b_map)
    out = {"a": name_a, "b": name_b, "n_sentences": len(gold_rows), "bootstrap": {}}
    for metric in ("trigger", "argument", "event"):
        out["bootstrap"][metric] = paired_bootstrap(ca, cb, metric=metric, n_boot=n_boot)
    out["mcnemar_event_exact"] = mcnemar(ca, cb)
    return out


def per_type_counts(gold_rows, pred_map, event_types):
    """返回 {type: {"counts": [...], "support": n}}，用于 per-type 表。"""
    res = {}
    for t in event_types:
        cs = sentence_counts(gold_rows, pred_map, type_filter=t)
        res[t] = {"counts": cs, "support": sum(x["n_gold"] for x in cs)}
    return res


def per_type_metrics(gold_rows, pred_map, event_types):
    res = {}
    for t in event_types:
        cs = sentence_counts(gold_rows, pred_map, type_filter=t)
        trig, _ = f1_from(cs, "trigger")
        arg, _ = f1_from(cs, "argument")
        ev, _ = f1_from(cs, "event")
        res[t] = {"support": sum(x["n_gold"] for x in cs),
                  "trigger_f1": round(trig, 4),
                  "argument_f1": round(arg, 4),
                  "event_f1": round(ev, 4)}
    return res


def write_significance(pairs, out_path, gold_rows):
    out = []
    for a, pa, b, pb in pairs:
        print(f"[sig] {a} vs {b} ...", flush=True)
        out.append(compare_pair(a, pa, b, pb, gold_rows))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[sig] → {out_path}")
    return out


if __name__ == "__main__":
    gold = load_jsonl(str(ROOT / "data/processed/dev_extract.jsonl"))
    a = sys.argv[1] if len(sys.argv) > 1 else "bert_crf_256"
    b = sys.argv[2] if len(sys.argv) > 2 else "roberta_crf_256"
    r = compare_pair(a, ROOT / f"outputs/disc/pred_{a}.jsonl",
                     b, ROOT / f"outputs/disc/pred_{b}.jsonl", gold)
    print(json.dumps(r, ensure_ascii=False, indent=2))
