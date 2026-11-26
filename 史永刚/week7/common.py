import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IGNORE = -100


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_schema(path=None):
    raw = json.loads(Path(path or DATA / "event_schema.json").read_text(encoding="utf-8"))
    return {k: list(v) for k, v in raw.items() if not k.startswith("__")}


def union_roles(schema):
    return sorted({r for roles in schema.values() for r in roles})


def make_labels(schema):
    tags = [f"TRG:{t}" for t in sorted(schema)] + [f"ARG:{r}" for r in union_roles(schema)]
    labels = ["O"] + [f"{p}-{t}" for t in tags for p in ("B", "I")]
    return labels, {x: i for i, x in enumerate(labels)}, {i: x for i, x in enumerate(labels)}


def row_event(row):
    return row.get("event") or (row.get("events") or [None])[0]


def span_to_tags(text, event, label2id):
    tags = ["O"] * len(text)

    def paint(s, e, tag):
        s, e = max(0, int(s)), min(len(text), int(e))
        if s >= e or f"B-{tag}" not in label2id:
            return
        tags[s] = f"B-{tag}"
        for i in range(s + 1, e):
            tags[i] = f"I-{tag}"

    for a in event.get("args") or []:
        paint(a.get("start", 0), a.get("end", 0), f"ARG:{a.get('role', '')}")
    trig = event.get("trigger") or {}
    if trig:
        paint(trig.get("start", 0), trig.get("end", 0), f"TRG:{event.get('type', '')}")
    return [label2id[t] for t in tags]


def decode_spans(tags):
    spans, cur = [], None
    for i, t in enumerate(tags):
        if t == "O":
            if cur:
                spans.append(cur)
                cur = None
        elif t.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = {"tag": t[2:], "start": i, "end": i + 1}
        else:
            tag = t[2:]
            if cur and cur["tag"] == tag:
                cur["end"] = i + 1
            else:
                if cur:
                    spans.append(cur)
                cur = {"tag": tag, "start": i, "end": i + 1}
    if cur:
        spans.append(cur)
    return spans


def spans_to_event(spans, text):
    trig = next((s for s in spans if s["tag"].startswith("TRG:")), None)
    if not trig:
        return None
    args = []
    for s in spans:
        if s["tag"].startswith("ARG:"):
            role = s["tag"][4:]
            if role:
                args.append({"role": role, "text": text[s["start"]:s["end"]]})
    return {"type": trig["tag"][4:],
            "trigger": {"text": text[trig["start"]:trig["end"]]},
            "args": args}


def normalize_event(e):
    trig = e.get("trigger") or {}
    trig = trig.get("text", "") if isinstance(trig, dict) else str(trig)
    args = Counter()
    for a in e.get("args") or e.get("arguments") or []:
        role = a.get("role", "")
        text = a.get("text") or a.get("argument") or ""
        if role and text:
            args[(role, text.strip())] += 1
    return {"type": e.get("type", ""), "trigger": str(trig).strip(), "args": args}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def compare_events(gold, pred):
    g_by, p_by = defaultdict(list), defaultdict(list)
    for e in gold:
        g_by[e["type"]].append(e)
    for e in pred:
        p_by[e["type"]].append(e)
    tr, ar, ev = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    for t in set(g_by) | set(p_by):
        gs, ps = g_by.get(t, []), p_by.get(t, [])
        for i in range(max(len(gs), len(ps))):
            g = gs[i] if i < len(gs) else None
            p = ps[i] if i < len(ps) else None
            if p is None:
                ev[2] += 1
                tr[2] += bool(g["trigger"])
                ar[2] += sum(g["args"].values())
                continue
            if g is None:
                ev[1] += 1
                tr[1] += bool(p["trigger"])
                ar[1] += sum(p["args"].values())
                continue
            if p["trigger"] and p["trigger"] == g["trigger"]:
                tr[0] += 1
            else:
                tr[1] += bool(p["trigger"])
                tr[2] += bool(g["trigger"])
            inter = sum((g["args"] & p["args"]).values())
            ar[0] += inter
            ar[1] += sum(p["args"].values()) - inter
            ar[2] += sum(g["args"].values()) - inter
            if p["trigger"] == g["trigger"] and g["args"] == p["args"]:
                ev[0] += 1
            else:
                ev[1] += 1
                ev[2] += 1
    return tr, ar, ev


def evaluate(gold_rows, pred_rows):
    pmap = {r.get("id", i): r for i, r in enumerate(pred_rows)}
    tr, ar, ev = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    for i, g in enumerate(gold_rows):
        p = pmap.get(g.get("id", i), {})
        gold = [normalize_event(x) for x in ([g["event"]] if g.get("event") else g.get("events") or [])]
        pred = [normalize_event(x) for x in p.get("events") or []]
        t, a, e = compare_events(gold, pred)
        tr = [x + y for x, y in zip(tr, t)]
        ar = [x + y for x, y in zip(ar, a)]
        ev = [x + y for x, y in zip(ev, e)]
    return {"trigger": prf(*tr), "argument": prf(*ar), "event": prf(*ev)}


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
