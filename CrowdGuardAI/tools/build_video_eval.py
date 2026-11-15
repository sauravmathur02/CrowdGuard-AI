"""
Download a labelled video evaluation set: Celeb-DF v2 official *test* list
(List_of_testing_videos.txt: "1 <path>" = real, "0 <path>" = fake), taken from the
Hugging Face mirror gonnerthetooner/deepfake_video_data (celebdf_v2/).

Only videos that are both in the official test list and present in the mirror are used.
Output: data/eval_cache/video/<subset>/<file>.mp4 and data/eval_cache/video/manifest.json
Usage:  python tools/build_video_eval.py [--per-class N]
"""
import argparse
import json
import os
import random

import requests

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(APP_DIR, "data", "eval_cache", "video")
REPO = "gonnerthetooner/deepfake_video_data"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/celebdf_v2/"
SEED = 1234


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    n = ap.parse_args().per_class
    os.makedirs(OUT, exist_ok=True)
    lines = requests.get(BASE + "List_of_testing_videos.txt", timeout=60).text.split()
    test = [(lines[i + 1], 1 - int(lines[i])) for i in range(0, len(lines), 2)]  # file label 1 = real
    present = set()
    for sub in ("Celeb-real", "YouTube-real", "Celeb-synthesis"):
        tree = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/main/celebdf_v2/{sub}", timeout=60).json()
        present |= {t["path"].split("celebdf_v2/", 1)[1] for t in tree}
    avail = [(p, y) for p, y in test if p in present]
    rng = random.Random(SEED)
    rng.shuffle(avail)
    real = [a for a in avail if a[1] == 0][:n]
    fake = [a for a in avail if a[1] == 1][:n]
    print(f"test list {len(test)}, in mirror {len(avail)} -> using {len(real)} real + {len(fake)} fake")
    items = []
    for i, (p, y) in enumerate(real + fake, 1):
        dst = os.path.join(OUT, p.replace("/", os.sep))
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            r = requests.get(BASE + p, timeout=180)
            r.raise_for_status()
            with open(dst, "wb") as f:
                f.write(r.content)
        items.append({"path": p, "label": y, "subset": p.split("/")[0]})
        if i % 20 == 0:
            print(f"  {i}/{len(real) + len(fake)}")
    json.dump({"source": f"huggingface.co/datasets/{REPO} (Celeb-DF v2 official test list)", "items": items},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
