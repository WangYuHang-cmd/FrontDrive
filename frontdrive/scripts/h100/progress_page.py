#!/usr/bin/env python3
"""Static HTML progress page for the H100 runs (stdlib only; run ON THE SERVER by scripts/h100/progress_web.sh every minute).

Reads: nvidia-smi, /proc/loadavg, free, df, tmux, the STOP/PAUSE/NOTE markers, every run dir under $EXP with a train.log that
changed in the last 2 days (progress line of the Lightning bar, params.json, accounting.json, exports, gpu_mem.csv),
every $EXP/eval_pb/*/report.json, the pb_smoke pick and the tail of state/phase5.log. Keeps a small history
(state/www/history.json) of the online-scorer validation proxy per run and epoch.
usage: progress_page.py --out state/www/index.html [--exp DIR] [--state DIR] [--refresh 60]
The page shows no addresses of other hosts and no command lines of other users' processes (only their process name).
"""
import argparse
import datetime as dt
import fcntl
import glob
import html
import json
import os
import re
import subprocess
import time

QD = "/home/henry/Desktop/Qwen_Drive"
OUR_ROOTS = ("/home/exouser/henry", QD)
PROG_RE = re.compile(r"Epoch (\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)(it/s|s/it)")
MET_RE = re.compile(r"((?:train|val)/[A-Za-z0-9_]+)=([-+]?[\d.]+(?:e[-+]?\d+)?)")


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(error: {type(e).__name__})"


def read_tail(path, nbytes=2_000_000):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - nbytes))
            return fh.read().decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return ""


def hms(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def parse_clock(s):
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def lock_held(path):
    """True when another process holds an flock on path (the chain's per-run lock). Never creates the file (an earlier
    version opened it in append mode and thereby planted lock files into idle run dirs, which kept them listed)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError:
        pass
    return False


def gpu_info():
    q = sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits")
    try:
        used, total, util, temp = [int(x) for x in q.split(",")]
    except ValueError:
        return {"used": None, "total": None, "util": None, "temp": None, "procs": [], "ours": 0}
    procs, ours = [], 0
    for line in sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits").splitlines():
        try:
            pid, mem = [x.strip() for x in line.split(",")]
            mem = int(mem)
        except ValueError:
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            cwd = ""
        mine = cwd.startswith(OUR_ROOTS)
        try:
            name = open(f"/proc/{pid}/comm").read().strip()
        except OSError:
            name = "?"
        if mine:
            ours += mem
        procs.append({"pid": pid, "mib": mem, "ours": mine, "name": name})
    return {"used": used, "total": total, "util": util, "temp": temp, "procs": procs, "ours": ours}


def run_info(run_dir, now):
    tag = os.path.basename(run_dir)
    info = {"tag": tag, "dir": run_dir}
    for name in ("params.json", "accounting.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                info[name.split(".")[0]] = json.load(open(p))
            except Exception:  # noqa: BLE001
                pass
    tl = os.path.join(run_dir, "train.log")
    info["log_mtime"] = os.path.getmtime(tl) if os.path.exists(tl) else None
    text = read_tail(tl)
    prog = None
    for m in PROG_RE.finditer(text):
        prog = m
    if prog:
        ep, pct, step, total, elapsed, eta, rate, unit = prog.groups()
        rate = float(rate)
        s_per_step = (1 / rate) if unit == "it/s" else rate
        info["progress"] = {"epoch": int(ep), "pct": int(pct), "step": int(step), "total": int(total), "elapsed": elapsed,
                            "eta_epoch": eta, "s_per_step": s_per_step}
        line = text[prog.start():text.find("\n", prog.start()) if text.find("\n", prog.start()) > 0 else None]
        mets = dict(MET_RE.findall(line))
        info["train_metrics"] = {k: float(v) for k, v in mets.items() if k.startswith("train/")}
        info["val_metrics"] = {k: float(v) for k, v in mets.items() if k.startswith("val/")}
    exports = sorted(glob.glob(os.path.join(run_dir, "exports", "epoch_*.ckpt")), key=lambda p: int(re.search(r"epoch_(\d+)", p)[1]))
    info["exports"] = [(int(re.search(r"epoch_(\d+)", p)[1]), os.path.getmtime(p)) for p in exports]
    info["last_ckpt"] = os.path.exists(os.path.join(run_dir, "checkpoints", "last.ckpt"))
    info["running"] = lock_held(os.path.join(run_dir, ".phase5.lock")) or (info["log_mtime"] is not None and now - info["log_mtime"] < 600)
    samples = []
    gm = os.path.join(run_dir, "gpu_mem.csv")
    if os.path.exists(gm):
        for line in read_tail(gm, 200_000).splitlines()[-120:]:
            p = line.split(",")
            if len(p) >= 4 and p[0].isdigit():
                try:
                    samples.append((int(p[0]), int(p[1]) if p[1].isdigit() else 0, int(p[2]) if p[2].isdigit() else 0, int(p[3]) if p[3].isdigit() else 0))
                except ValueError:
                    pass
    info["gpu_samples"] = samples
    # ETA for the whole run: remaining steps of this epoch + remaining epochs * (steps * s/step + validation ~3 min)
    epochs = (info.get("params") or {}).get("epochs")
    if prog and epochs:
        pr = info["progress"]
        remaining_this = (pr["total"] - pr["step"]) * pr["s_per_step"]
        per_epoch = pr["total"] * pr["s_per_step"] + 180
        info["eta_run_s"] = remaining_this + max(0, epochs - pr["epoch"] - 1) * per_epoch
    return info


def sparkline(samples, key_idx, vmax, width=360, height=48, color="#4c9be8"):
    if len(samples) < 2:
        return ""
    pts = []
    n = len(samples)
    for i, s in enumerate(samples):
        x = i * (width - 2) / (n - 1) + 1
        y = height - 1 - (min(s[key_idx], vmax) / vmax) * (height - 2)
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="#f4f6f8"/>'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{" ".join(pts)}"/></svg>')


def fmt_ts(ts, tz):
    return dt.datetime.fromtimestamp(ts, tz).strftime("%m-%d %H:%M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp", default=f"{QD}/benchmarks/navsim/exp")
    ap.add_argument("--state", default=f"{QD}/state")
    ap.add_argument("--refresh", type=int, default=60)
    a = ap.parse_args()
    now = time.time()
    utc, edt = dt.timezone.utc, dt.timezone(dt.timedelta(hours=-4), "EDT")
    gpu = gpu_info()
    load = open("/proc/loadavg").read().split()[:3] if os.path.exists("/proc/loadavg") else ["?"] * 3
    ram = sh("free -g | awk 'NR==2{print $7\" GB avail of \"$2}'")
    disk = sh("df -h / | awk 'NR==2{print $4\" free\"}'")
    up_h = int(float(open("/proc/uptime").read().split()[0]) // 3600) if os.path.exists("/proc/uptime") else "?"
    tmux = sh("tmux ls 2>/dev/null | cut -d: -f1 | tr '\\n' ' '")
    markers = " ".join(m for m in ("STOP", "PAUSE") if os.path.exists(os.path.join(a.state, m))) or "none"
    note = open(os.path.join(a.state, "NOTE")).read()[:200] if os.path.exists(os.path.join(a.state, "NOTE")) else ""

    runs = []
    for d in sorted(glob.glob(os.path.join(a.exp, "*"))):
        tl = os.path.join(d, "train.log")
        # active or recent runs only: log written in the last 24 h, or a chain lock present (pushed-in stale E1 dirs drop out)
        if os.path.isdir(d) and os.path.exists(tl) and os.path.basename(d) != "pb_smoke" and \
                (now - os.path.getmtime(tl) < 86400 or os.path.exists(os.path.join(d, ".phase5.lock"))):
            runs.append(run_info(d, now))
    # validation-proxy history per run (epoch -> val metrics), persisted next to the page
    hist_p = os.path.join(os.path.dirname(a.out), "history.json")
    try:
        hist = json.load(open(hist_p))
    except Exception:  # noqa: BLE001
        hist = {}
    for r in runs:
        vm = r.get("val_metrics") or {}
        pr = r.get("progress")
        if vm.get("val/score_epoch") is not None and pr:
            h = hist.setdefault(r["tag"], {})
            key = str(pr["epoch"])  # validation of epoch N-1 is visible while epoch N (0-based) runs -> completed epochs = N
            if key not in h:
                h[key] = {"completed_epochs": pr["epoch"], "utc": int(now), **{k.replace("val/", ""): v for k, v in vm.items()}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(hist, open(hist_p + ".tmp", "w"), indent=1)
    os.replace(hist_p + ".tmp", hist_p)

    evals = []
    for rep in sorted(glob.glob(os.path.join(a.exp, "eval_pb", "*", "report.json"))):
        try:
            d = json.load(open(rep))
            d["_mtime"] = os.path.getmtime(rep)
            evals.append(d)
        except Exception:  # noqa: BLE001
            pass
    ref = next((e for e in evals if e.get("tag") == "pb_official_v1_dev"), None)
    pick = None
    try:
        pick = json.load(open(os.path.join(a.exp, "pb_smoke", "pick.json"))).get("baseline_recommendation")
    except Exception:  # noqa: BLE001
        pass
    chain_tail = read_tail(os.path.join(a.state, "phase5.log"), 60_000).splitlines()[-18:]

    e = html.escape
    out = []
    out.append(f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta http-equiv="refresh" content="{a.refresh}">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>H100 训练进度</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px;color:#1b1f23;background:#fff}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:22px 0 6px;border-bottom:1px solid #e1e4e8;padding-bottom:3px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #e1e4e8;padding:3px 7px;text-align:left;vertical-align:top}}
th{{background:#f6f8fa}} .ok{{color:#1a7f37}} .warn{{color:#b35900}} .bad{{color:#c62828}} .mono{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}}
.bar{{background:#e9ecef;border-radius:3px;height:12px;width:260px;display:inline-block;vertical-align:middle}} .bar>i{{display:block;height:12px;border-radius:3px;background:#4c9be8}}
pre{{background:#f6f8fa;padding:8px;font-size:12px;overflow-x:auto;white-space:pre-wrap}} .muted{{color:#6a737d;font-size:12px}}</style></head><body>
<h1>H100 训练进度 <span class="muted">生成于 {dt.datetime.fromtimestamp(now, utc).strftime('%Y-%m-%d %H:%M:%S')} UTC / {dt.datetime.fromtimestamp(now, edt).strftime('%H:%M:%S')} EDT，每 {a.refresh} s 自动刷新</span></h1>""")
    used, total, util = gpu["used"], gpu["total"], gpu["util"]
    free_gb = (total - used) / 1024 if used is not None else None
    cls = "ok" if (free_gb is None or free_gb >= 20) else "bad"
    out.append(f"""<p class="mono">GPU {used} / {total} MiB（我们 {gpu['ours']} MiB，<span class="{cls}">空闲 {free_gb:.1f} GB</span>）· 利用率 {util}% · {gpu['temp']}°C · load {' '.join(load)} · RAM {e(ram)} · 磁盘 {e(disk)} · up {up_h} h<br>
tmux: {e(tmux) or '—'} · 标记: <b>{markers}</b>{(' · NOTE: ' + e(note)) if note else ''}</p>""" if used is not None else "<p class='bad'>nvidia-smi 不可用</p>")
    if gpu["procs"]:
        out.append("<table><tr><th>pid</th><th>MiB</th><th>归属</th><th>进程</th></tr>" + "".join(
            f"<tr><td class='mono'>{p['pid']}</td><td>{p['mib']}</td><td>{'ours' if p['ours'] else '<b>FOREIGN</b>'}</td><td>{e(p['name'])}</td></tr>" for p in gpu["procs"]) + "</table>")

    out.append("<h2>训练任务</h2>")
    if not runs:
        out.append("<p class='muted'>最近 2 天没有活动的训练目录</p>")
    for r in runs:
        pr, params, acc = r.get("progress"), r.get("params") or {}, r.get("accounting")
        state = "<span class='ok'>运行中</span>" if r["running"] else ("<span class='ok'>已完成</span>" if acc and acc.get("rc") == 0 else "<span class='warn'>已停止 / 未在跑</span>")
        out.append(f"<h3 style='margin:14px 0 4px;font-size:15px'>{e(r['tag'])} {state}</h3>")
        meta = []
        if params:
            meta.append(f"batch {params.get('batch')}×{params.get('accum')} · epochs {params.get('epochs')} · act-ckpt {'off' if params.get('encoder_activation_checkpointing') is False else 'on'} · 初始化 {e(os.path.basename(str(params.get('pretrain_pt', ''))))} · seed {params.get('seed')}")
        if pr:
            epochs = params.get("epochs") or "?"
            pct_run = (pr["epoch"] + pr["step"] / pr["total"]) / epochs * 100 if isinstance(epochs, int) else None
            eta_run = dt.datetime.fromtimestamp(now + r["eta_run_s"], edt).strftime("%m-%d %H:%M EDT") if r.get("eta_run_s") is not None else "?"
            meta.append(f"epoch {pr['epoch'] + 1}/{epochs}：第 {pr['step']}/{pr['total']} 步（{pr['pct']}%，本 epoch 还需 {pr['eta_epoch']}）· {pr['s_per_step']:.2f} s/步 · 全程 ETA ≈ {eta_run}")
            if pct_run is not None:
                meta.append(f"<span class='bar'><i style='width:{pct_run:.1f}%'></i></span> {pct_run:.1f}%")
        if r["gpu_samples"]:
            last = r["gpu_samples"][-1]
            peak = max(s[1] for s in r["gpu_samples"])
            util_mean = sum(s[3] for s in r["gpu_samples"]) / len(r["gpu_samples"])
            meta.append(f"显存（本进程）{last[1]} MiB，近 1 h 峰值 {peak} MiB · GPU 利用率近 1 h 均值 {util_mean:.0f}%<br>{sparkline(r['gpu_samples'], 3, 100)} <span class='muted'>GPU 利用率 %（近 {len(r['gpu_samples'])} 个 30 s 采样）</span>")
        if r["exports"]:
            meta.append("导出: " + ", ".join(f"ep{n} ({fmt_ts(ts, edt)})" for n, ts in r["exports"]))
        if acc:
            meta.append(f"accounting: rc {acc.get('rc')} · wall {hms(acc.get('wall_s'))} · 峰值 {acc.get('peak_vram_mib_our_pid')} MiB · {acc.get('s_per_step_last_progress') and round(acc['s_per_step_last_progress'], 2)} s/步")
        out.append("<p class='mono'>" + "<br>".join(meta) + "</p>")
        tm = r.get("train_metrics") or {}
        if tm:
            keys = [k for k in ("train/loss", "train/trajectory_loss", "train/final_score_loss", "train/sub_score_loss", "train/score", "train/best_score") if k in tm]
            out.append("<p class='mono muted'>train: " + " · ".join(f"{k.split('/')[1]} {tm[k]:.3f}" for k in keys) + "</p>")
        h = hist.get(r["tag"]) or {}
        if h:
            rows = sorted(h.values(), key=lambda x: x["completed_epochs"])
            out.append("<table><tr><th>已完成 epoch</th><th>val score（在线打分器代理）</th><th>collision</th><th>dac</th><th>progress</th><th>ttc</th><th>comfort</th><th>l2</th></tr>" + "".join(
                f"<tr><td>{x['completed_epochs']}</td><td><b>{x.get('score_epoch', float('nan')):.3f}</b></td><td>{x.get('collision', float('nan')):.3f}</td><td>{x.get('dac', float('nan')):.3f}</td><td>{x.get('progress', float('nan')):.3f}</td><td>{x.get('ttc', float('nan')):.3f}</td><td>{x.get('comfort', float('nan')):.3f}</td><td>{x.get('l2', float('nan')):.2f}</td></tr>" for x in rows) + "</table>")

    # V-JEPA pretraining arms (benchmarks/pretrain/runs/<run>/log_r0.csv + accounting.json), active in the last 24 h
    pre_root = os.path.join(QD, "benchmarks", "pretrain", "runs")
    arms = []
    for d in sorted(glob.glob(os.path.join(pre_root, "*"))):
        lg = os.path.join(d, "log_r0.csv")
        if not os.path.exists(lg) or now - os.path.getmtime(lg) > 86400:
            continue
        last = read_tail(lg, 20_000).strip().splitlines()
        row = last[-1].split(",") if len(last) > 1 else []
        acct = {}
        try:
            acct = json.load(open(os.path.join(d, "accounting.json")))
        except Exception:  # noqa: BLE001
            pass
        peak = 0
        gm = os.path.join(d, "gpu_mem.csv")
        if os.path.exists(gm):
            for line in read_tail(gm, 100_000).splitlines():
                p = line.split(",")
                if len(p) >= 2 and p[1].isdigit():
                    peak = max(peak, int(p[1]))
        arms.append({"run": os.path.basename(d), "mtime": os.path.getmtime(lg), "row": row, "acct": acct, "peak": peak,
                     "alive": now - os.path.getmtime(lg) < 900})
    if arms:
        out.append("<h2>V-JEPA 预训练臂（benchmarks/pretrain/runs）</h2><table><tr><th>run</th><th>状态</th><th>epoch,itr</th><th>loss</th><th>iter / gpu / data ms</th><th>tokens/步</th><th>累计步数（accounting）</th><th>exit_status</th><th>峰值显存</th><th>最近写入</th></tr>")
        for arm in arms:  # NB: not `a` — that is the argparse namespace used further down (bug of 10-20 03:50Z broke the page for 9 h)
            r = arm["row"]; ep_itr = f"{r[0]},{r[1]}" if len(r) > 1 else "?"; loss = r[2] if len(r) > 2 else "?"
            ms = f"{r[7]} / {r[8]} / {r[9]}" if len(r) > 9 else "?"; tok = r[10] if len(r) > 10 else "?"
            st = "<span class='ok'>运行中</span>" if arm["alive"] else f"<span class='muted'>{e(str(arm['acct'].get('exit_status', '停止')))}</span>"
            out.append(f"<tr><td class='mono'>{e(arm['run'])}</td><td>{st}</td><td>{e(ep_itr)}</td><td>{e(loss)}</td><td>{e(ms)}</td><td>{e(tok)}</td><td>{arm['acct'].get('steps', '')}</td><td>{e(str(arm['acct'].get('exit_status', '')))}</td><td>{arm['peak'] or ''}</td><td class='muted'>{fmt_ts(arm['mtime'], edt)}</td></tr>")
        out.append("</table><p class='muted'>epoch = 2,500 步；40 epoch = 100k 步；loss 为 V-JEPA 目标（加运动头时含分项）。</p>")
    out.append("<h2>dev 评估（navtrain 官方 val，18,157 场景；正式 PDMS）</h2>")
    if evals:
        out.append("<table><tr><th>评估</th><th>split</th><th>PDMS</th><th>NC</th><th>DAC</th><th>EP</th><th>TTC</th><th>C</th><th>DDC</th><th>Δ vs 官方 pb dev 参照</th><th>完成</th></tr>")
        for d in sorted(evals, key=lambda x: (x.get("split", ""), x.get("tag", ""))):
            delta = ""
            if ref and d.get("split") == "navtrain_val_official" and d is not ref and d.get("pdms") is not None:
                dv = d["pdms"] - ref["pdms"]
                delta = f"<span class='{'ok' if dv >= 0 else 'warn'}'>{dv:+.2f}</span>"
            out.append(f"<tr><td class='mono'>{e(d.get('tag', '?'))}</td><td>{e(d.get('split', ''))}</td><td><b>{d.get('pdms')}</b></td><td>{d.get('no_at_fault_collisions')}</td><td>{d.get('drivable_area_compliance')}</td><td>{d.get('ego_progress')}</td><td>{d.get('time_to_collision_within_bound')}</td><td>{d.get('comfort')}</td><td>{d.get('driving_direction_compliance')}</td><td>{delta}</td><td class='muted'>{fmt_ts(d['_mtime'], edt)}</td></tr>")
        out.append("</table>")
        if ref:
            out.append(f"<p class='muted'>参照 = 官方 pb v1 权重在 dev 上的 {ref['pdms']}（同一权重 navtest 93.69）。② 的 G-B 判据：自训基线 late 均值 ≥ {ref['pdms'] - 0.5:.2f}。</p>")
    else:
        out.append("<p class='muted'>还没有评估报告</p>")
    # perception-free dev evals (scripts/eval_perception_free.sh -> exp/eval/<tag>/report.json), recent ones only
    pf_evals = []
    for rep in sorted(glob.glob(os.path.join(a.exp, "eval", "*", "report.json"))):
        try:
            if now - os.path.getmtime(rep) > 7 * 86400:
                continue
            d = json.load(open(rep))
            d["_mtime"] = os.path.getmtime(rep)
            pf_evals.append(d)
        except Exception:  # noqa: BLE001
            pass
    if pf_evals:
        out.append("<h2>perception-free dev 评估（最近 7 天，exp/eval）</h2><table><tr><th>评估</th><th>split</th><th>PDMS</th><th>NC</th><th>DAC</th><th>EP</th><th>TTC</th><th>C</th><th>DDC</th><th>完成</th></tr>")
        for d in sorted(pf_evals, key=lambda x: x.get("tag", "")):
            out.append(f"<tr><td class='mono'>{e(d.get('tag', '?'))}</td><td>{e(str(d.get('split', '')))}</td><td><b>{d.get('score', d.get('pdms'))}</b><td>{d.get('no_at_fault_collisions')}</td><td>{d.get('drivable_area_compliance')}</td><td>{d.get('ego_progress')}</td><td>{d.get('time_to_collision_within_bound')}</td><td>{d.get('comfort')}</td><td>{d.get('driving_direction_compliance')}</td><td class='muted'>{fmt_ts(d['_mtime'], edt)}</td></tr>")
        out.append("</table><p class='muted'>pf 参照：官方 perception-free 权重 dev 87.11（本机 87.10）；E1 比较看 late 窗口均值（C-29）。</p>")
    if pick:
        out.append(f"<p class='muted'>冒烟推荐（pb_smoke/pick.json）：BATCH {pick.get('BATCH')} × ACCUM {pick.get('ACCUM')}，ACT_CKPT {pick.get('ACT_CKPT')}，峰值 {pick.get('peak_vram_mib')} MiB，{pick.get('s_per_step') and round(pick['s_per_step'], 2)} s/步（30 步冒烟）</p>")
    out.append("<h2>链日志（state/phase5.log 末尾）</h2><pre>" + e("\n".join(line[:260] for line in chain_tail)) + "</pre>")
    out.append("<p class='muted'>页面由 scripts/h100/progress_page.py 生成；只读；不显示其他用户进程的命令行。</p></body></html>")
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(out))
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
