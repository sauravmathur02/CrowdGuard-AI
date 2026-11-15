"""
Download a small, labelled evaluation set for calibrating the deepfake detectors.

Sources (labels come from each dataset's own metadata / folder structure):
  1. Defactify (Hugging Face: Rajarshi-Roy-research/Defactify_Image_Dataset), validation split.
     Label_A: 0 = real (MS-COCO photo), 1 = AI-generated.
     Label_B: 0 = real, 1 = SD 2.1, 2 = SDXL, 3 = SD 3, 4 = DALL-E 3, 5 = Midjourney.
  2. DeepFakeFace / DFF (Hugging Face: OpenRL/DeepFakeFace).
     wiki.zip = real IMDB-WIKI faces; insight.zip = InsightFace face swaps;
     inpainting.zip = SD inpainting; text2img.zip = SD 1.5 text-to-image faces.
     Only the needed members are fetched with HTTP range requests (no full 1 GB zips).
  3. GenD sample crops (GitHub: yermandy/deepfake-detection, datasets/): FF++ and Celeb-DF v2.

Output: CrowdGuardAI/data/eval_cache/<source>/<subset>/<file> and manifest.json
Usage:  python tools/build_eval_set.py [--per-class N]
"""
import argparse
import io
import json
import os
import random
import sys
import time
import zipfile

import requests

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(APP_DIR, "data", "eval_cache")
SEED = 1234
S = requests.Session()
S.headers["User-Agent"] = "CrowdGuardAI-calibration/1.0"


def _get(url, **kw):
    for attempt in range(5):
        try:
            r = S.get(url, timeout=60, **kw)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url = url
        h = S.head(url, allow_redirects=True, timeout=60)
        h.raise_for_status()
        self.url = h.url
        self.size = int(h.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        data = _get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}).content
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def fetch_defactify(per_class_real, per_generator):
    ds = "Rajarshi-Roy-research/Defactify_Image_Dataset"
    gen_names = {1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "midjourney"}
    want = {0: per_class_real, **{k: per_generator for k in gen_names}}
    got = {k: 0 for k in want}
    items = []
    rng = random.Random(SEED)
    offsets = list(range(0, 9000, 100))
    rng.shuffle(offsets)
    for off in offsets:
        if all(got[k] >= want[k] for k in want):
            break
        rows = _get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": ds, "config": "default", "split": "validation", "offset": off, "length": 100},
        ).json()["rows"]
        for row in rows:
            r = row["row"]
            b = int(r["Label_B"])
            if b not in want or got[b] >= want[b]:
                continue
            sub = "real" if b == 0 else gen_names[b]
            src = r["Image"]["src"]
            ext = ".jpg" if ".jpg" in src.split("?")[0].lower() or ".jpeg" in src.lower() else ".png"
            rel = os.path.join("defactify", sub, f"{off + row['row_idx']:05d}{ext}")
            path = os.path.join(OUT_DIR, rel)
            if not os.path.exists(path):
                try:
                    _save(path, _get(src).content)
                except Exception as e:
                    print(f"  skip {rel}: {e}")
                    continue
            got[b] += 1
            items.append({"path": rel, "label": int(r["Label_A"]), "source": "defactify", "subset": sub, "face": False})
        print(f"  defactify offset {off}: {got}")
    return items


def fetch_dff(per_subset_real, per_subset_fake):
    base = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/"
    plan = [("wiki", 0, per_subset_real), ("insight", 1, per_subset_fake),
            ("inpainting", 1, per_subset_fake), ("text2img", 1, per_subset_fake)]
    items = []
    rng = random.Random(SEED)
    for name, label, n in plan:
        print(f"  DFF {name}.zip: reading central directory...")
        zf = zipfile.ZipFile(HTTPRangeFile(base + f"{name}.zip"))
        members = [m for m in zf.infolist() if not m.is_dir() and m.filename.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(members)
        for m in members[:n]:
            rel = os.path.join("dff", name, m.filename.replace("/", "_"))
            path = os.path.join(OUT_DIR, rel)
            if not os.path.exists(path):
                _save(path, zf.read(m))
            items.append({"path": rel, "label": label, "source": "dff", "subset": name, "face": True})
        print(f"  DFF {name}: {min(n, len(members))} of {len(members)} images")
    return items


def fetch_gend_samples():
    tree = _get("https://api.github.com/repos/yermandy/deepfake-detection/git/trees/main?recursive=1").json()["tree"]
    items = []
    for t in tree:
        p = t["path"]
        if t["type"] != "blob" or not p.startswith("datasets/") or not p.lower().endswith((".png", ".jpg")):
            continue
        parts = p.split("/")  # datasets/<ds>/<subset>/<video>/<frame>
        subset = parts[2]
        label = 0 if subset in ("real", "Celeb-real", "YouTube-real") else 1
        rel = os.path.join("gend_samples", *parts[1:])
        path = os.path.join(OUT_DIR, rel)
        if not os.path.exists(path):
            _save(path, _get(f"https://raw.githubusercontent.com/yermandy/deepfake-detection/main/{p}").content)
        items.append({"path": rel, "label": label, "source": "gend_samples", "subset": f"{parts[1]}/{subset}", "face": True})
    print(f"  GenD samples: {len(items)} crops")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=250, help="real images per source")
    args = ap.parse_args()
    n = args.per_class
    os.makedirs(OUT_DIR, exist_ok=True)
    items = []
    print("[1/3] Defactify (general AI-generated vs real photos)")
    items += fetch_defactify(per_class_real=n, per_generator=max(1, n // 5))
    print("[2/3] DeepFakeFace (face swaps / inpainting / generated faces vs real faces)")
    items += fetch_dff(per_subset_real=n, per_subset_fake=max(1, n // 3))
    print("[3/3] GenD sample crops (FF++, Celeb-DF v2)")
    items += fetch_gend_samples()
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "items": items}, f, indent=1)
    by = {}
    for it in items:
        by.setdefault(f"{it['source']}/{it['subset']}", 0)
        by[f"{it['source']}/{it['subset']}"] += 1
    print(json.dumps(by, indent=1))
    print(f"Total {len(items)} images -> {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
