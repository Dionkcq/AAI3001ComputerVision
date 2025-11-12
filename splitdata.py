# build_od_split_40_10_two_pools.py
import os, sys, re, csv, shutil, random, argparse
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
    import imagehash
    PHASH_AVAILABLE = True
except Exception:
    PHASH_AVAILABLE = False

IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",
              ".JPG",".JPEG",".PNG",".BMP",".TIF",".TIFF",".WEBP"}

DEFAULT_CLASS_MAP = {
    "bench press":     ("cz_benchpress",     "benchpress"),
    "deadlift":        ("cz_deadlift",       "deadlift"),
    "leg extension":   ("cz_legextension",   "leg_extension"),
    "push up":         ("cz_pushup",         "pushup"),
    "shoulder press":  ("cz_shoulderpress",  "shoulder_press"),
    "squat":           ("cz_squat",          "squat"),
}

def parse_args():
    ap = argparse.ArgumentParser(
        description="OD split: 40 train (Dataset), 10 val (Dataset_cz), leftovers to separate pools"
    )
    ap.add_argument("--root", required=True, help="Folder containing 'Dataset/' and 'Dataset_cz/'")
    ap.add_argument("--out", required=True, help="Output root for new dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_val", type=int, default=10)
    ap.add_argument("--angle_keywords", default="left,right,front,back")
    ap.add_argument("--min_per_angle_train", type=int, default=8)
    ap.add_argument("--min_per_angle_val", type=int, default=3)
    ap.add_argument("--phash_dist", type=int, default=10,
                    help="0 to disable (needs Pillow+imagehash if >0)")
    ap.add_argument("--stem_guard", action="store_true",
                    help="Avoid same filename stem across train/val")
    return ap.parse_args()

def list_images(d: Path):
    return [p for p in d.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]

def angle_of(path: Path, tokens):
    parts = [s.lower() for s in Path(path).parts]
    for t in tokens:
        rx = re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
        if any(rx.search(p) for p in parts):
            return t
    return "other"

def stratified_pick(imgs, target, tokens, min_per_angle, rng):
    buckets = defaultdict(list)
    for p in imgs:
        buckets[angle_of(p, tokens)].append(p)
    for k in buckets:
        rng.shuffle(buckets[k])

    picked = []
    # pass 1: angle minimums
    for k in list(buckets.keys()):
        take = min(min_per_angle, len(buckets[k]))
        for _ in range(take):
            if len(picked) >= target: break
            picked.append(buckets[k].pop())
        if len(picked) >= target: break
    # pass 2: round-robin
    while len(picked) < target and any(buckets.values()):
        for k in list(buckets.keys()):
            if buckets[k]:
                picked.append(buckets[k].pop())
                if len(picked) >= target:
                    break
    return picked

def compute_phash(p: Path):
    with Image.open(p) as im:
        return imagehash.phash(im)

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    root = Path(args.root)
    out_root = Path(args.out)

    A = root / "Dataset"
    B = root / "Dataset_cz"
    if not A.exists() or not B.exists():
        sys.exit("Both 'Dataset/' and 'Dataset_cz/' must exist under --root")

    tokens = [t.strip().lower() for t in args.angle_keywords.split(",") if t.strip()]

    # output layout
    for sub in [
        "images/train", "images/val",
        "pool_dataset", "pool_dataset_cz"
    ]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    manifest_rows = []  # split,class,src,dst,angle,source_pool
    totals = {"train":0, "val":0, "pool_dataset":0, "pool_dataset_cz":0}

    # for leakage control between train and val only
    used_hashes_train, used_hashes_val = [], []
    used_stems_train,  used_stems_val  = set(), set()
    if args.phash_dist > 0 and not PHASH_AVAILABLE:
        print("[WARN] imagehash not available; pHash dedup disabled.")

    def ok_for_val(p):
        if args.stem_guard and p.stem in used_stems_train:
            return False
        if args.phash_dist > 0 and PHASH_AVAILABLE:
            try:
                h = compute_phash(p)
            except Exception:
                return True
            if any(abs(h - ht) <= args.phash_dist for ht in used_hashes_train):
                return False
        return True

    def ok_for_train(p):
        if args.stem_guard and p.stem in used_stems_val:
            return False
        if args.phash_dist > 0 and PHASH_AVAILABLE:
            try:
                h = compute_phash(p)
            except Exception:
                return True
            if any(abs(h - hv) <= args.phash_dist for hv in used_hashes_val):
                return False
        return True

    for src_cls, (cz_cls, out_cls) in DEFAULT_CLASS_MAP.items():
        a_dir = A / src_cls
        b_dir = B / cz_cls

        imgs_a = list_images(a_dir) if a_dir.exists() else []
        imgs_b = list_images(b_dir) if b_dir.exists() else []

        rng.shuffle(imgs_a); rng.shuffle(imgs_b)

        pick_train = stratified_pick(imgs_a, min(args.n_train, len(imgs_a)),
                                     tokens, args.min_per_angle_train, rng)
        pick_val   = stratified_pick(imgs_b, min(args.n_val, len(imgs_b)),
                                     tokens, args.min_per_angle_val, rng)

        # enforce dedup between train & val only
        final_train = [p for p in pick_train if ok_for_train(p)]
        if len(final_train) < args.n_train:
            for p in imgs_a:
                if p not in final_train and p not in pick_train and ok_for_train(p):
                    final_train.append(p)
                if len(final_train) >= args.n_train:
                    break

        if args.phash_dist > 0 and PHASH_AVAILABLE:
            for p in final_train:
                try: used_hashes_train.append(compute_phash(p))
                except Exception: pass
        if args.stem_guard:
            used_stems_train |= {p.stem for p in final_train}

        final_val = [p for p in pick_val if ok_for_val(p)]
        if len(final_val) < args.n_val:
            for p in imgs_b:
                if p not in final_val and p not in pick_val and ok_for_val(p):
                    final_val.append(p)
                if len(final_val) >= args.n_val:
                    break

        if args.phash_dist > 0 and PHASH_AVAILABLE:
            for p in final_val:
                try: used_hashes_val.append(compute_phash(p))
                except Exception: pass
        if args.stem_guard:
            used_stems_val |= {p.stem for p in final_val}

        # destinations
        train_dir = out_root / "images/train" / out_cls
        val_dir   = out_root / "images/val"   / out_cls
        poolA_dir = out_root / "pool_dataset" / out_cls
        poolB_dir = out_root / "pool_dataset_cz" / out_cls
        for d in (train_dir, val_dir, poolA_dir, poolB_dir):
            d.mkdir(parents=True, exist_ok=True)

        chosen_train = set(final_train)
        chosen_val   = set(final_val)

        # copy train
        for p in final_train:
            dst = train_dir / p.name
            shutil.copy2(p, dst)
            manifest_rows.append(["train", out_cls, str(p), str(dst), angle_of(p, tokens), ""])
            totals["train"] += 1

        # copy val
        for p in final_val:
            dst = val_dir / p.name
            shutil.copy2(p, dst)
            manifest_rows.append(["val", out_cls, str(p), str(dst), angle_of(p, tokens), ""])
            totals["val"] += 1

        # leftovers split into two pools (separate)
        leftovers_A = [p for p in imgs_a if p not in chosen_train]
        leftovers_B = [p for p in imgs_b if p not in chosen_val]

        for p in leftovers_A:
            dst = poolA_dir / p.name
            shutil.copy2(p, dst)
            manifest_rows.append(["pool_dataset", out_cls, str(p), str(dst), angle_of(p, tokens), "A"])
            totals["pool_dataset"] += 1

        for p in leftovers_B:
            dst = poolB_dir / p.name
            shutil.copy2(p, dst)
            manifest_rows.append(["pool_dataset_cz", out_cls, str(p), str(dst), angle_of(p, tokens), "B"])
            totals["pool_dataset_cz"] += 1

        print(f"{out_cls:<16} -> train {len(final_train):>2}/{args.n_train} | "
              f"val {len(final_val):>2}/{args.n_val} | "
              f"poolA +{len(leftovers_A)} | poolB +{len(leftovers_B)}")


    for k,v in totals.items():
        print(f"{k:>17}: {v}")
    print("Out folder:", out_root)

if __name__ == "__main__":
    main()
