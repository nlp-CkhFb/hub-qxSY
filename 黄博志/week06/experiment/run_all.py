"""
一键跑完所有对比方法。

用法：
  # 正式实验（约 1.5–2 小时 CPU，可后台跑）
  python run_all.py --methods all

  # 冒烟测试（小样本、5 步，约 2–3 分钟）
  python run_all.py --smoke --methods all --force

  # 只重跑某几个方法
  python run_all.py --methods bert_ft,roberta_ft --force

输出：
  outputs/<method>/metrics.json
  outputs/<method>/predictions.json
  outputs/summary.md
  outputs/summary.csv
"""

import argparse
import json
import time

import numpy as np

import config
import data
import metrics


def _parse_methods(s: str):
    if s.strip().lower() == "all":
        return list(config.METHODS.keys())
    methods = [m.strip() for m in s.split(",") if m.strip()]
    unknown = [m for m in methods if m not in config.METHODS]
    if unknown:
        raise SystemExit(f"未知方法：{unknown}；可选：{list(config.METHODS)}")
    return methods


def _run_one(method, train, dev, test, args):
    if method == "rule":
        import baselines
        return baselines.run_rule(train, test)
    if method == "tfidf_svc":
        import baselines
        return baselines.run_tfidf(train, test, kind="svc")
    if method == "tfidf_nb":
        import baselines
        return baselines.run_tfidf(train, test, kind="nb")
    if method in ("bert_frozen", "roberta_frozen"):
        import embed_lr
        kind = method.split("_")[0]
        return embed_lr.run_frozen(kind, train, test, pool=args.pool, force=args.force)
    if method in ("bert_ft", "roberta_ft"):
        import finetune
        kind = method.split("_")[0]
        return finetune.run_finetune(
            kind,
            train,
            dev,
            test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_len=args.max_len,
            max_steps=args.max_steps,
            seed=args.seed,
            save_model=args.save_model,
        )
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    parser.add_argument("--force-split", action="store_true", help="重新做数据切分")
    parser.add_argument("--smoke", action="store_true", help="小样本冒烟测试")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--max-len", type=int, default=config.MAX_LEN)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        config.OUT_DIR = config.ROOT / "outputs_smoke"
        config.FEATURE_DIR = config.OUT_DIR / "features"
        args.epochs = 1
        args.max_steps = args.max_steps or 5
        args.batch_size = min(args.batch_size, 16)
        print("[smoke] 输出目录：", config.OUT_DIR)

    config.ensure_dirs()
    train, dev, test, meta, dist = data.load_splits(force=args.force_split)

    if args.smoke:
        train = data._stratified_sample(train, 300, args.seed)
        dev = data._stratified_sample(dev, 100, args.seed + 1)
        test = data._stratified_sample(test, 300, args.seed + 2)
        print(f"[smoke] train={len(train)} dev={len(dev)} test={len(test)}")

    methods = _parse_methods(args.methods)
    print("方法：", methods)
    print("数据：", json.dumps(meta, ensure_ascii=False))

    y_true = np.array([r["label"] for r in test], dtype=np.int64)
    summary = []

    for method in methods:
        method_dir = config.OUT_DIR / method
        metrics_path = method_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"[skip] {method} 已有结果（--force 可重跑）")
            with open(metrics_path, "r", encoding="utf-8") as f:
                summary.append(json.load(f))
            continue

        method_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== {method} / {config.METHODS[method]} =====", flush=True)
        t0 = time.perf_counter()
        out = _run_one(method, train, dev, test, args)
        wall = time.perf_counter() - t0

        result = metrics.evaluate(
            y_true, out["pred"], out.get("prob"), labels=range(config.NUM_LABELS)
        )
        result.update(
            {
                "method": method,
                "display_name": config.METHODS[method],
                "test_n": len(test),
                "train_seconds": round(float(out["train_seconds"]), 3),
                "infer_seconds": round(float(out["infer_seconds"]), 3),
                "infer_ms_per_sample": round(
                    float(out["infer_seconds"]) / max(1, len(test)) * 1000.0, 3
                ),
                "wall_seconds": round(wall, 3),
                "extra": out.get("extra", {}),
            }
        )

        metrics.save_json(metrics_path, result)

        with open(method_dir / "predictions.json", "w", encoding="utf-8") as f:
            rows = []
            for r, p in zip(test, out["pred"]):
                row = {
                    "uid": r["uid"],
                    "sentence": r["sentence"],
                    "true": int(r["label"]),
                    "pred": int(p),
                }
                if out.get("prob") is not None:
                    row["prob"] = [round(float(x), 6) for x in out["prob"][len(rows)]]
                rows.append(row)
            json.dump(rows, f, ensure_ascii=False)

        summary.append(result)
        print(
            f"[done] {method}: acc={result['acc']:.4f} "
            f"macro-F1={result['macro_f1']:.4f} "
            f"train={result['train_seconds']:.1f}s "
            f"infer={result['infer_ms_per_sample']:.2f}ms/条",
            flush=True,
        )

    if not args.no_report:
        import report
        report.build_report(config.OUT_DIR)


if __name__ == "__main__":
    main()
