"""
Evaluate (and optionally fit) the video verdict on Celeb-DF v2 test videos
(tools/build_video_eval.py). Run after tools/calibrate.py.

  python tools/eval_video.py           # score (cached), fit "video_face" fusion on the fit half, report on the test half
  python tools/eval_video.py --no-fit  # report only, using the image-fitted fusion for videos

Production path that is measured:
  * 16 frames sampled evenly (app.VIDEO_FRAMES) -> calibrated per-frame scores ->
    mean of the most suspicious half of frames (ensemble.aggregate_frames)
  * GenD worker on the same video: most suspicious face per frame, same aggregation
  * fusion: "video_face" model when a face was scored (fitted here), else the image "no_face" model
Celeb-DF contains face swaps only, so the fitted video model is specific to face-swap video.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "tools"))
os.chdir(APP_DIR)

from detectors import calibration  # noqa: E402
from calibrate import _logit, auc, fit_logreg, old_auto_rule, rates  # noqa: E402

VDIR = os.path.join(APP_DIR, "data", "eval_cache", "video")
SCORES = os.path.join(VDIR, "scores.jsonl")
N_FRAMES = 16


def split_of(path):
    return "fit" if int(hashlib.sha1(path.encode()).hexdigest(), 16) % 2 == 0 else "test"


def score_all(items):
    done = {}
    if os.path.exists(SCORES):
        for line in open(SCORES, encoding="utf-8"):
            r = json.loads(line)
            done[r["path"]] = r
    todo = [it for it in items if it["path"] not in done]
    if not todo:
        return list(done.values())
    calibration._LOADED_CALIB = calibration.DEFAULT_CALIB  # raw scores while scoring
    import cv2
    import torch
    from detectors.pixel import BFreeDetector, CLIPDetector, CommForDetector
    from detectors.video import sample_video_bytes_with_timestamps
    import gend_worker

    commfor, bfree, clip = CommForDetector(), BFreeDetector(), CLIPDetector()
    old = None
    od = os.path.join(APP_DIR, "pretrained_weights", "deepfake_face")
    if os.environ.get("CG_OLD_FACE") and os.path.isdir(od):  # optional, slow
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        old = (AutoImageProcessor.from_pretrained(od), AutoModelForImageClassification.from_pretrained(od).to(gend_worker.DEVICE).eval())

    t0 = time.time()
    with open(SCORES, "a", encoding="utf-8") as out:
        for n, it in enumerate(todo, 1):
            data = open(os.path.join(VDIR, it["path"].replace("/", os.sep)), "rb").read()
            frames, _ = sample_video_bytes_with_timestamps(data, n_frames=N_FRAMES)
            c = [float(x) for x in commfor.predict(frames)]
            b = [None if x is None else float(x) for x in bfree.predict(frames)] if bfree.is_available else []
            k = [None if x is None else float(x) for x in clip.predict(frames)] if clip.is_available else []
            face = gend_worker.score_media(data, os.path.basename(it["path"]))
            old_frames = []
            if old is not None:
                for f in gend_worker.video_frames(data, ".mp4", n=12):  # old worker sampled 12 frames
                    crops = [cr for cr, _ in gend_worker.face_crops(f)]
                    if crops:
                        with torch.inference_mode():
                            inp = old[0](images=crops, return_tensors="pt").to(gend_worker.DEVICE)
                            old_frames.append(float(torch.softmax(old[1](**inp).logits, -1)[:, 1].max()))
            rec = {**it, "commfor_f": c, "bfree_f": b, "clip_f": k, "gend_raw": face.get("gend"),
                   "gend_f": face.get("frame_scores", []), "faces": face.get("faces", 0), "box": face.get("box"),
                   "old_face_f": old_frames}
            out.write(json.dumps(rec) + "\n")
            done[it["path"]] = rec
            if n % 10 == 0:
                out.flush()
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s")
    return list(done.values())


def new_scores(r):
    from detectors import ensemble
    cal = lambda d, xs: [calibration.calibrate_score(d, x) for x in xs if x is not None]
    s = {
        "commfor": ensemble.aggregate_frames(cal("commfor", r["commfor_f"])),
        "bfree": ensemble.aggregate_frames(cal("bfree", r["bfree_f"])),
        "clip": ensemble.aggregate_frames(cal("clip", r["clip_f"])),
        "gend": calibration.calibrate_score("gend", r["gend_raw"]) if r["gend_raw"] is not None else None,
    }
    return s, ensemble.fuse(s, media="video")["p_fake"]


def old_verdict(r):
    """Approximation of the previous video path: 8 frames, trimmed-mean pixel scores,
    old face ViT with the old worker's video aggregation, old hand-tuned rules (is_video)."""
    def tm(xs):
        xs = [x for x in xs if x is not None][::2]  # ~8 of the 16 frames
        if not xs:
            return None
        if len(xs) >= 5:
            s = sorted(xs)
            return float(0.55 * np.mean(s[1:-1]) + 0.45 * np.median(xs))
        return float(np.mean(xs))
    p = r["old_face_f"]
    g = None
    if len(p) >= 4:
        a = np.array(p); med, mean, ratio = float(np.median(a)), float(a.mean()), float((a > 0.5).mean())
        if ratio >= 0.40 and med > 0.45:
            g = 0.60 * med + 0.40 * float(np.mean(sorted(p, reverse=True)[:max(2, len(p) // 2)]))
        elif ratio >= 0.25 and med > 0.35:
            g = 0.5 * med + 0.5 * mean
        else:
            g = 0.85 * med + 0.15 * mean
    elif len(p) >= 2:
        g = float(np.median(p))
    rr = {"commfor": tm(r["commfor_f"]), "bfree": tm(r["bfree_f"]), "clip": (float(np.mean([x for x in r["clip_f"] if x is not None])) if r["clip_f"] else None),
          "gend_old": g, "faces": r["faces"], "box": None}
    return old_auto_rule(rr, is_video=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fit", action="store_true")
    args = ap.parse_args()
    man = json.load(open(os.path.join(VDIR, "manifest.json")))
    rows = score_all(man["items"])
    calibration.reload()

    if not args.no_fit:
        from detectors import ensemble  # noqa: F401
        fit = [r for r in rows if split_of(r["path"]) == "fit" and r["gend_raw"] is not None]
        feats = ["commfor", "bfree", "clip", "gend"]
        X, y = [], []
        for r in fit:
            s, _ = new_scores(r)
            if all(s[f] is not None for f in feats):
                X.append([_logit(s[f]) for f in feats]); y.append(r["label"])
        w, b = fit_logreg(np.array(X), y)
        cal = json.load(open(calibration.CALIB_PATH, encoding="utf-8"))
        cal["fusion"]["video_face"] = {"features": feats, "w": w, "b": b, "n_fit": len(y),
                                       "data": man["source"] + " (fit half)"}
        json.dump(cal, open(calibration.CALIB_PATH, "w", encoding="utf-8"), indent=2)
        calibration.reload()
        print(f"Fitted video_face fusion on {len(y)} videos: w={[round(x, 3) for x in w]} b={b:+.3f}")

    test = [r for r in rows if split_of(r["path"]) == "test"]
    thr = float(calibration.load().get("threshold", 0.5))
    y = [r["label"] for r in test]
    newp = [new_scores(r)[1] for r in test]
    res = {
        "n_test": len(test),
        "old_rules": rates([old_verdict(r) for r in test], y),
        "new_fusion": rates([p > thr for p in newp], y),
        "new_auc": auc(newp, y),
    }
    det = {}
    for d in ("commfor", "bfree", "clip", "gend"):
        s = [(new_scores(r)[0][d], r["label"]) for r in test if new_scores(r)[0][d] is not None]
        if s:
            det[d] = {"auc": auc([a for a, _ in s], [b for _, b in s]), "n": len(s)}
    res["single_detectors_auc"] = det
    res["faces_scored"] = sum(1 for r in test if r["gend_raw"] is not None)
    f = lambda m: f"fake caught {m['fake_caught'] * 100:.0f}%, real correct {m['real_correct'] * 100:.0f}%, balanced {m['balanced_acc'] * 100:.0f}%"
    print(f"\nCeleb-DF v2 test half: {len(test)} videos ({sum(y)} fake), faces scored in {res['faces_scored']}")
    print("  OLD :", f(res["old_rules"]))
    print("  NEW :", f(res["new_fusion"]), f"| AUC {res['new_auc']:.3f}")
    print("  single-detector AUC:", {k: round(v["auc"], 3) for k, v in det.items()})
    cal = json.load(open(calibration.CALIB_PATH, encoding="utf-8"))
    cal["eval_video"] = {"data": man["source"], **res}
    json.dump(cal, open(calibration.CALIB_PATH, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
