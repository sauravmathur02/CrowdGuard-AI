"""
Score the labelled evaluation set (tools/build_eval_set.py) with every detector, fit the
calibration + fusion used by the live API, and report held-out accuracy.

  python tools/calibrate.py            # score (cached) + fit + report, writes data/calibration.json
  python tools/calibrate.py --report   # only report using the existing calibration.json

What is fitted (on the "fit" half, evaluated on the untouched "test" half):
  * Platt scaling per detector:   p_cal = sigmoid(a * logit(p_raw) + b)   (class-balanced)
  * Fusion logistic regression on calibrated logits, two variants:
      no_face: commfor, bfree(NPR), clip
      face:    commfor, bfree(NPR), clip, gend      (rows where a face was scored)
Each image is scored twice: as downloaded, and re-saved as JPEG quality 85 (typical of
images shared online). Both copies of an image always land in the same split.
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

from detectors import calibration  # noqa: E402

EVAL_DIR = os.path.join(APP_DIR, "data", "eval_cache")
SCORES_PATH = os.path.join(EVAL_DIR, "scores.jsonl")
EVAL_SCORES_OUT = os.path.join(APP_DIR, "data", "eval_scores.json")
FEATURES_NO_FACE = ["commfor", "bfree", "clip"]
FEATURES_FACE = ["commfor", "bfree", "clip", "gend"]
VARIANTS = ("orig", "jpeg85")


# ── scoring ────────────────────────────────────────────────────────────────
def score_all(items):
    done = {}
    if os.path.exists(SCORES_PATH):
        with open(SCORES_PATH, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[(r["path"], r["variant"])] = r
    todo = [(it, v) for it in items for v in VARIANTS if (it["path"], v) not in done]
    if not todo:
        return list(done.values())

    print(f"Scoring {len(todo)} image variants ({len(done)} cached)...")
    # Raw scores: identity calibration while scoring.
    calibration._LOADED_CALIB = calibration.DEFAULT_CALIB
    from detectors.pixel import BFreeDetector, CLIPDetector, CommForDetector
    from detectors import ensemble
    import gend_worker

    commfor, bfree, clip = CommForDetector(), BFreeDetector(), CLIPDetector()

    # The previous face model, scored on the same crops, to compare old vs new behaviour.
    old_face = None
    old_dir = os.path.join(APP_DIR, "pretrained_weights", "deepfake_face")
    if os.environ.get("CG_OLD_FACE") and os.path.isdir(old_dir):  # optional, slow
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        old_face = (AutoImageProcessor.from_pretrained(old_dir),
                    AutoModelForImageClassification.from_pretrained(old_dir).to(gend_worker.DEVICE).eval())

    t0 = time.time()
    with open(SCORES_PATH, "a", encoding="utf-8") as out:
        for n, (it, variant) in enumerate(todo, 1):
            path = os.path.join(EVAL_DIR, it["path"])
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"  skip {it['path']}: {e}")
                continue
            if variant == "jpeg85":
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=85)
                data = buf.getvalue()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                name = "x.jpg"
            else:
                with open(path, "rb") as f:
                    data = f.read()
                name = os.path.basename(path)

            face = gend_worker.score_media(data, name)
            s = ensemble.score_image(img, commfor, bfree, clip, face_box=face.get("box"))
            old_g = None
            if old_face is not None and face.get("gend") is not None:
                import cv2
                bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                crops = [c for c, _ in gend_worker.face_crops(bgr)]
                if crops:
                    with torch.inference_mode():
                        inp = old_face[0](images=crops, return_tensors="pt").to(gend_worker.DEVICE)
                        old_g = float(torch.softmax(old_face[1](**inp).logits, -1)[:, 1].max())
            rec = {**it, "variant": variant, **s, "gend": face.get("gend"), "gend_old": old_g,
                   "faces": face.get("faces", 0), "box": face.get("box")}
            out.write(json.dumps(rec) + "\n")
            done[(it["path"], variant)] = rec
            if n % 100 == 0:
                out.flush()
                print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s")
    return list(done.values())


# ── fitting ────────────────────────────────────────────────────────────────
def _logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_logreg(X, y, l2=1e-2, iters=300):
    """Class-balanced L2 logistic regression. Returns (w, b)."""
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    pos = y.sum().clamp(min=1)
    neg = (1 - y).sum().clamp(min=1)
    sw = torch.where(y > 0.5, 0.5 / pos, 0.5 / neg)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).sum() + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy().tolist(), float(b.detach())


def split_of(path):
    return "fit" if int(hashlib.sha1(path.encode()).hexdigest(), 16) % 2 == 0 else "test"


def fit_calibration(rows):
    fit = [r for r in rows if split_of(r["path"]) == "fit"]
    platt = {}
    for d in FEATURES_FACE:
        rr = [r for r in fit if r.get(d) is not None]
        (a,), b = fit_logreg(_logit([r[d] for r in rr])[:, None], [r["label"] for r in rr], l2=1e-4)
        platt[d] = [a, b]
        print(f"  Platt {d:8s} a={a:+.3f} b={b:+.3f}  (n={len(rr)})")

    def cal(r, d):
        a, b = platt[d]
        return float(1 / (1 + np.exp(-(a * _logit(r[d]) + b))))

    fusion = {}
    for name, feats, sel in (("no_face", FEATURES_NO_FACE, lambda r: True),
                             ("face", FEATURES_FACE, lambda r: r.get("gend") is not None)):
        rr = [r for r in fit if sel(r) and all(r.get(f) is not None for f in feats)]
        X = np.array([[_logit(cal(r, f)) for f in feats] for r in rr])
        w, b = fit_logreg(X, [r["label"] for r in rr])
        fusion[name] = {"features": feats, "w": w, "b": b, "n_fit": len(rr)}
        print(f"  Fusion {name:8s} n={len(rr)} w={[round(x, 3) for x in w]} b={b:+.3f}")
    return platt, fusion


# ── evaluation ─────────────────────────────────────────────────────────────
def auc(scores, y):
    s, y = np.asarray(scores, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rates(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, int)
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else None
    tnr = float((~pred[y == 0]).mean()) if (y == 0).any() else None
    bal = None if tpr is None or tnr is None else (tpr + tnr) / 2
    return {"n": int(len(y)), "fake_caught": tpr, "real_correct": tnr, "balanced_acc": bal, "acc": float((pred == (y == 1)).mean())}


def old_auto_rule(r, is_video=False):
    """The previous hand-tuned 'auto' verdict from app.py (raw scores, old face model, no
    compression gating, ELA override or temporal term). Kept only to compare before/after."""
    c, b, k, g = r["commfor"], r["bfree"], r["clip"], r.get("gend_old")
    faces, box = r.get("faces", 0), r.get("box")
    area = box[2] * box[3] if box else 0.0
    large, anyf = faces > 0 and area > 0.05, faces > 0 and area > 0.02
    if large and g is not None:
        wg, wb, wc, wk = 0.45, 0.35, 0.15, 0.05
    elif anyf and g is not None:
        wg, wb, wc, wk = 0.35, 0.30, 0.25, 0.10
    else:
        wc, wk, wb, wg = 0.50, 0.30, 0.20, 0.0
    act = [(c, wc)] + ([(b, wb)] if b is not None else []) + ([(k, wk)] if k is not None and wk > 0 else [])
    if g is not None and faces > 0 and wg > 0:
        act.append((g, wg))
    pf = sum(s * w for s, w in act) / (sum(w for _, w in act) or 1)
    thr = 0.40
    pix = [c] + ([b] if b is not None else []) + ([k] if k is not None else [])
    if len(pix) >= 2 and sum(s > 0.5 for s in pix) >= 2:
        pf, thr = max(pf, max(pix) * 0.92), 0.45
    elif g is not None and g > 0.70 and large:
        pf, thr = max(pf, g * 0.92), 0.45
    elif g is not None and g > 0.65 and faces > 0:
        pf, thr = max(pf, g * 0.90), 0.48
    elif max(pix) > 0.88:
        if min(pix) < 0.12 and not large and not is_video:
            pf = 0.5 * max(pix) + 0.5 * min(pix)
        else:
            pf, thr = max(pf, max(pix) * 0.85), 0.45
    elif len(pix) >= 2 and max(pix) > 0.55 and min(pix) < 0.15 and not large and not is_video:
        pf, thr = min(pf, 0.36), 0.40
    else:
        thr = 0.48
    return pf > thr


def report(rows):
    from detectors import ensemble
    calibration.reload()
    cal = calibration.load()
    thr = float(cal.get("threshold", 0.5))
    test = [r for r in rows if split_of(r["path"]) == "test"]

    def c(r, d):
        return calibration.calibrate_score(d, r[d]) if r.get(d) is not None else None

    for r in test:
        r["_new"] = ensemble.fuse({d: c(r, d) for d in FEATURES_FACE})["p_fake"]
        r["_old"] = old_auto_rule(r)

    groups = {"ALL": test}
    for r in test:
        groups.setdefault(f"{r['source']}/{r['subset']}", []).append(r)
        groups.setdefault(f"variant={r['variant']}", []).append(r)
        groups.setdefault("faces scored" if r.get("gend") is not None else "no face", []).append(r)
    src_groups = {}
    for r in test:
        src_groups.setdefault(f"source={r['source']}", []).append(r)
    groups.update(src_groups)

    out = {}
    print(f"\nHeld-out test split: {len(test)} image variants, threshold {thr}")
    hdr = f"{'group':28s} {'n':>5s} | {'OLD fake/real/bal':>18s} | {'NEW fake/real/bal':>18s} | NEW AUC"
    print(hdr)
    print("-" * len(hdr))

    def fmt(m):
        f = lambda v: "  - " if v is None else f"{v * 100:3.0f}%"
        return f"{f(m['fake_caught'])} {f(m['real_correct'])} {f(m['balanced_acc'])}"

    for g, rr in groups.items():
        y = [r["label"] for r in rr]
        old = rates([r["_old"] for r in rr], y)
        new = rates([r["_new"] > thr for r in rr], y)
        a = auc([r["_new"] for r in rr], y)
        out[g] = {"old_rules": old, "new_fusion": new, "new_auc": a}
        print(f"{g:28s} {len(rr):5d} | {fmt(old):>18s} | {fmt(new):>18s} | {'-' if a is None else f'{a:.3f}'}")

    print("\nSingle detectors on the test split (calibrated, threshold 0.5):")
    det = {}
    for d in FEATURES_FACE + ["gend_old"]:
        rr = [r for r in test if r.get(d) is not None]
        if not rr:
            continue
        y = [r["label"] for r in rr]
        s = [c(r, d) if d != "gend_old" else r[d] for r in rr]
        m = rates([v > 0.5 for v in s], y)
        m["auc"] = auc(s, y)
        det[d] = m
        print(f"  {d:9s} n={m['n']:4d}  AUC={m['auc']:.3f}  fake={m['fake_caught'] * 100:3.0f}%  real={m['real_correct'] * 100:3.0f}%")
    out["single_detectors"] = det
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="only report with the existing calibration.json")
    args = ap.parse_args()
    manifest = json.load(open(os.path.join(EVAL_DIR, "manifest.json"), encoding="utf-8"))
    rows = score_all(manifest["items"])

    if not args.report:
        print("\nFitting calibration on the fit split...")
        platt, fusion = fit_calibration(rows)
        counts = {}
        for r in rows:
            k = f"{r['source']}/{r['subset']}"
            counts[k] = counts.get(k, 0) + 1
        calib = {
            "version": 2,
            "fitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "temperature": dict(calibration.DEFAULT_CALIB["temperature"]),
            "platt": {**calibration.DEFAULT_CALIB["platt"], **platt,
                      "npr": platt["bfree"], "npr_resnet": platt["bfree"]},
            "fusion": fusion,
            "threshold": 0.5,
            "data": {
                "sources": {
                    "defactify": "huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset (validation)",
                    "dff": "huggingface.co/datasets/OpenRL/DeepFakeFace",
                    "gend_samples": "github.com/yermandy/deepfake-detection (datasets/)",
                },
                "image_variants": list(VARIANTS),
                "counts": counts,
                "split": "sha1(path) parity: even=fit, odd=test",
            },
        }
        with open(calibration.CALIB_PATH, "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
        print(f"Wrote {calibration.CALIB_PATH}")

    metrics = report(rows)
    cal = json.load(open(calibration.CALIB_PATH, encoding="utf-8"))
    cal["eval"] = {"split": "test", **metrics}
    with open(calibration.CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    keep = ["path", "source", "subset", "label", "variant", "commfor", "bfree", "clip", "gend", "gend_old", "faces"]
    with open(EVAL_SCORES_OUT, "w", encoding="utf-8") as f:
        json.dump([{k: r.get(k) for k in keep} for r in rows], f)
    print(f"Wrote held-out metrics into calibration.json and raw scores to {EVAL_SCORES_OUT}")


if __name__ == "__main__":
    main()
