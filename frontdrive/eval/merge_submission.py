#!/usr/bin/env python3
"""Merge the per-seed NAVSIM submission pickles (run_create_submission_pickle.py, one per seed) into ONE multi-seed submission and
write the team metadata (plan v4.3 ⑤ / G-D). The leaderboard reads `predictions` as a list of {token: Trajectory} dicts, one per
seed, and reports max / mean / std over them; nothing here selects a seed.
usage: merge_submission.py --out submission.pkl --team-name "..." --authors "..." --email "..." --institution "..." --country "..."
       seed0/submission.pkl seed1/submission.pkl [seed2/submission.pkl ...]
Checks: identical token sets across seeds, 12,146 navtest tokens, every pose array finite with shape (8, 3)."""
import argparse
import pickle
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--team-name", required=True)
    ap.add_argument("--authors", required=True)
    ap.add_argument("--email", required=True)
    ap.add_argument("--institution", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--expect-tokens", type=int, default=12146)
    ap.add_argument("pickles", nargs="+")
    a = ap.parse_args()
    preds = []
    for p in a.pickles:
        d = pickle.load(open(p, "rb"))
        assert isinstance(d.get("predictions"), list) and len(d["predictions"]) == 1, f"{p}: expected one prediction dict"
        preds.append(d["predictions"][0])
    tokens = set(preds[0])
    for p, pr in zip(a.pickles, preds):
        assert set(pr) == tokens, f"{p}: token set differs from the first pickle"
        assert len(pr) == a.expect_tokens, f"{p}: {len(pr)} tokens, expected {a.expect_tokens}"
        for tok, traj in pr.items():
            poses = np.asarray(traj.poses)
            assert poses.shape == (8, 3) and np.isfinite(poses).all(), f"{p}: bad trajectory for {tok}: {poses.shape}"
    sub = {"team_name": a.team_name, "authors": a.authors, "email": a.email, "institution": a.institution,
           "country / region": a.country, "predictions": preds}
    with open(a.out, "wb") as fh:
        pickle.dump(sub, fh)
    print(f"wrote {a.out}: {len(preds)} seeds x {len(tokens)} tokens; team_name={a.team_name!r}")


if __name__ == "__main__":
    sys.exit(main())
