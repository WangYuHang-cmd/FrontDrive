"""Build the V-JEPA clip list ("<mp4> 0" per line) and a frozen manifest for a comma clip directory (plan v4 §4.4).

Usage: python -m pretrain.make_clip_list --clips <dir> --out <list.csv> [--manifest <manifest.csv>] [--min-frames 100]
                                          [--require-can] [--exclude-ids <file with clip ids>]
Only clip ids (HMAC) appear in the outputs; the raw manifest with route ids stays outside the repo.
"""
import argparse, csv, glob, hashlib, json, os, sys
import av, numpy as np


def n_frames(mp4: str) -> int:
    try:
        with av.open(mp4) as c:
            st = c.streams.video[0]
            return int(st.frames or 0) or sum(1 for _ in c.decode(st))
    except Exception:
        return -1


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="write clip_id, n_frames, has_can, device, bytes, sha256 (no route ids)")
    ap.add_argument("--min-frames", type=int, default=100); ap.add_argument("--require-can", action="store_true")
    ap.add_argument("--exclude-ids", default=None, help="text file of clip ids to drop (e.g. internal dongles)")
    ap.add_argument("--fps", type=float, default=2.0)
    a = ap.parse_args()
    excl = set(open(a.exclude_ids).read().split()) if a.exclude_ids else set()
    rows, kept, drop = [], [], {"short": 0, "bad": 0, "nocan": 0, "excluded": 0}
    for mp4 in sorted(glob.glob(os.path.join(a.clips, "*", "*.mp4"))):
        cid = os.path.basename(mp4)[:-4]
        if cid in excl:
            drop["excluded"] += 1; continue
        n = n_frames(mp4)
        if n < 0:
            drop["bad"] += 1; continue
        if n < a.min_frames:
            drop["short"] += 1; continue
        can = mp4[:-4] + ".can.npz"; has_can = os.path.exists(can)
        if a.require_can and not has_can:
            drop["nocan"] += 1; continue
        dev = ""
        if has_can:
            try:
                dev = str(np.load(can, allow_pickle=False)["device"])
            except Exception:
                has_can = False
        rows.append({"clip_id": cid, "n_frames": n, "has_can": int(has_can), "device": dev, "bytes": os.path.getsize(mp4),
                     "sha256": sha256(mp4) if a.manifest else ""})
        kept.append(mp4)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for p in kept:
            f.write(f"{p} 0\n")
    if a.manifest:
        with open(a.manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    hours = sum(r["n_frames"] for r in rows) / a.fps / 3600
    devs = {}
    for r in rows:
        devs[r["device"] or "unknown"] = devs.get(r["device"] or "unknown", 0) + 1
    summary = {"n_clips": len(kept), "hours": round(hours, 2), "with_can": sum(r["has_can"] for r in rows), "devices": devs,
               "dropped": drop, "gb": round(sum(r["bytes"] for r in rows) / 1e9, 2), "list": a.out,
               "list_sha256": sha256(a.out)}
    print(json.dumps(summary)); return 0


if __name__ == "__main__":
    sys.exit(main())
