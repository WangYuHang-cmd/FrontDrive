"""Per-scene pick analysis for the ④(c) inference-time ensembles (docs/evidence/2026-10-17_pb_baseline/README.md):
smoke check (mode `first` must reproduce member 0 exactly), which member argmax_conf selected per scene (inferred from exact
score equality), hit rate where members differ, oracle-of-M, and what adding the constant-lr third member cost.
Reads the pulled per-scene CSVs under benchmarks/navsim/exp/eval_pb/<tag>/ (scripts/h100/pull.sh).
Run: source env/drivejepa_env.sh && python eval/ens_pick_analysis.py"""
import glob, sys
import numpy as np, pandas as pd
E = "/home/henry/Desktop/Qwen_Drive/benchmarks/navsim/exp/eval_pb"
def load(tag):
    csv = sorted(glob.glob(f"{E}/{tag}/*.csv"))[-1]
    df = pd.read_csv(csv, index_col=0)
    df = df[df["token"].astype(str) != "average"].set_index("token")
    return df
sub = ["score","no_at_fault_collisions","drivable_area_compliance","ego_progress","time_to_collision_within_bound","comfort","driving_direction_compliance"]
m0, m1, b0 = load("pb_e50_cos_s0_soup16_20"), load("pb_e50_cos_s1_soup16_20"), load("pb_e50_s0_soup16_20")
ea, em, sm = load("pb_ens_cos01_argmax"), load("pb_ens_cos01_meanclose"), load("ens_smoke_cos01_first")
print("n scenes:", len(m0), len(m1), len(ea), len(em), "smoke", len(sm))
# 1) smoke (mode first) must reproduce member 0 exactly
j = sm.join(m0[sub], rsuffix="_m0", how="inner")
d = np.abs(j[sub].values.astype(float) - j[[c+"_m0" for c in sub]].values.astype(float))
print(f"SMOKE vs cos_s0 soup: {len(j)} tokens matched, max |diff| over 7 columns = {d.max():.3e}, exact-equal scenes = {(d.max(axis=1)==0).sum()}/{len(j)}")
# 2) means
for n, df in [("cos_s0 soup", m0), ("cos_s1 soup", m1), ("e50_s0 soup", b0), ("ens argmax", ea), ("ens meanclose", em)]:
    print(f"{n:14s} PDMS {df['score'].mean()*100:.2f}")
idx = m0.index.intersection(m1.index).intersection(ea.index).intersection(em.index)
s0, s1, sa, smc = (df.loc[idx, "score"].values.astype(float) for df in (m0, m1, ea, em))
print(f"oracle-of-2 (max per scene) {np.maximum(s0,s1).mean()*100:.2f}; mean-of-2 {((s0+s1)/2).mean()*100:.2f}")
# 3) which member did argmax pick? infer from exact score equality
eq0, eq1 = np.isclose(sa, s0, atol=1e-9), np.isclose(sa, s1, atol=1e-9)
diff = ~np.isclose(s0, s1, atol=1e-9)
print(f"members differ in score on {diff.sum()} / {len(idx)} scenes ({diff.mean()*100:.1f}%)")
print(f"argmax: ==m0 only {(eq0&~eq1).sum()}, ==m1 only {(eq1&~eq0).sum()}, ==both {(eq0&eq1).sum()}, ==neither {(~eq0&~eq1).sum()}")
dd = diff
better0 = s0 > s1
pick0 = eq0 & ~eq1; pick1 = eq1 & ~eq0
right = ((pick0 & better0) | (pick1 & ~better0)) & dd
wrong = ((pick0 & ~better0) | (pick1 & better0)) & dd
print(f"where members differ: picked better {right.sum()}, picked worse {wrong.sum()}, unresolved {(dd & ~right & ~wrong).sum()} -> hit rate {right.sum()/(right.sum()+wrong.sum())*100:.1f}% (chance 50%)")
gain = (sa - s0); print(f"argmax - cos_s0 soup: mean {gain.mean()*100:+.2f}, scenes better {(gain>1e-9).sum()}, worse {(gain<-1e-9).sum()}")
# 4) argmax vs meanclose
dm = ~np.isclose(sa, smc, atol=1e-9)
print(f"meanclose differs from argmax on {dm.sum()} scenes; mean diff {(smc-sa).mean()*100:+.3f}; better {((smc-sa)>1e-9).sum()}, worse {((smc-sa)<-1e-9).sum()}")
# 5) sub-scores
for n, df in [("cos_s0 soup", m0), ("cos_s1 soup", m1), ("ens argmax", ea), ("ens meanclose", em)]:
    print(n.ljust(14), " ".join(f"{c.split('_')[0][:4]}={df.loc[idx,c].mean()*100:.2f}" for c in sub[1:]))

print("\n==== three-member ensemble (pb_ens_cos01e50_argmax) ====")
import glob
import numpy as np, pandas as pd
E = "/home/henry/Desktop/Qwen_Drive/benchmarks/navsim/exp/eval_pb"
def load(tag):
    csv = sorted(glob.glob(f"{E}/{tag}/*.csv"))[-1]
    df = pd.read_csv(csv, index_col=0)
    return df[df["token"].astype(str) != "average"].set_index("token")
m0, m1, b0 = load("pb_e50_cos_s0_soup16_20"), load("pb_e50_cos_s1_soup16_20"), load("pb_e50_s0_soup16_20")
e2, e3 = load("pb_ens_cos01_argmax"), load("pb_ens_cos01e50_argmax")
idx = m0.index.intersection(m1.index).intersection(b0.index).intersection(e2.index).intersection(e3.index)
s = {n: df.loc[idx, "score"].values.astype(float) for n, df in [("m0", m0), ("m1", m1), ("b0", b0), ("e2", e2), ("e3", e3)]}
print("n", len(idx), " ".join(f"{k}={v.mean()*100:.2f}" for k, v in s.items()))
M = np.stack([s["m0"], s["m1"], s["b0"]], axis=1)
print(f"oracle-of-3 {M.max(axis=1).mean()*100:.2f}, oracle-of-2(cos) {np.maximum(s['m0'], s['m1']).mean()*100:.2f}, random-of-3 {M.mean(axis=1).mean()*100:.2f}")
eq = np.stack([np.isclose(s["e3"], M[:, i], atol=1e-9) for i in range(3)], axis=1)
only = eq & (eq.sum(axis=1, keepdims=True) == 1)
print(f"3-member argmax: ==m0 only {only[:,0].sum()}, ==m1 only {only[:,1].sum()}, ==e50_s0 only {only[:,2].sum()}, ambiguous(>=2 equal) {(eq.sum(axis=1)>=2).sum()}, none {(eq.sum(axis=1)==0).sum()}")
differ = ~(np.isclose(M[:,0], M[:,1], atol=1e-9) & np.isclose(M[:,0], M[:,2], atol=1e-9))
best = M.max(axis=1); worst = M.min(axis=1)
hit = np.isclose(s["e3"], best, atol=1e-9) & differ
miss_worst = np.isclose(s["e3"], worst, atol=1e-9) & differ & ~np.isclose(best, worst, atol=1e-9)
print(f"scenes where members differ {differ.sum()} ({differ.mean()*100:.1f}%): picked a best-scoring member {hit.sum()} ({hit.sum()/differ.sum()*100:.1f}%), picked the worst {miss_worst.sum()} ({miss_worst.sum()/differ.sum()*100:.1f}%)")
# how often did adding e50_s0 change the pick relative to the 2-member ensemble, and what did it cost
changed = ~np.isclose(s["e3"], s["e2"], atol=1e-9)
print(f"3-member differs from 2-member on {changed.sum()} scenes; mean diff {(s['e3']-s['e2']).mean()*100:+.2f}; better {((s['e3']-s['e2'])>1e-9).sum()}, worse {((s['e3']-s['e2'])<-1e-9).sum()}")
sel_b0 = only[:, 2]
print(f"when it switched to e50_s0 (n={sel_b0.sum()}): e50_s0 score mean {M[sel_b0,2].mean()*100:.2f} vs 2-member pick {s['e2'][sel_b0].mean()*100:.2f} vs best-of-cos {np.maximum(M[sel_b0,0],M[sel_b0,1]).mean()*100:.2f}")
sub = ["no_at_fault_collisions","drivable_area_compliance","ego_progress","time_to_collision_within_bound","comfort","driving_direction_compliance"]
for n, df in [("e50_s0 soup", b0), ("ens2 argmax", e2), ("ens3 argmax", e3)]:
    print(n.ljust(12), " ".join(f"{c.split('_')[0][:4]}={df.loc[idx,c].mean()*100:.2f}" for c in sub))
