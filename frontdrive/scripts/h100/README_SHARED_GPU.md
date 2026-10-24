# Shared-machine note / 共用说明（please read / 请先读）

**EN**
- Henry's research jobs run under this same account, in the tmux session `fdq` (list: `tmux ls`; look
  without touching: `tmux attach -r -t fdq`). All of our files are under `/home/exouser/henry/`
  (plus a symlink `/home/henry/Desktop/Qwen_Drive` that points into it) — please do not delete, move
  or edit anything there. Server clock is UTC.
- GPU: Henry's jobs have priority on this card and may use all of its memory (Henry, 2026-10-18). When you
  need the GPU, use the marker files below (PAUSE / STOP) or leave a NOTE and we free it within a minute;
  we never kill your processes. Our processes are the `python` ones whose working directory
  is under `/home/exouser/henry` (`ls -l /proc/<pid>/cwd`); they never touch yours.
- Need more GPU / CPU / RAM? Use marker files — no need to kill anything:
  - `touch /home/exouser/henry/Qwen_Drive/state/PAUSE` — we launch nothing new; running jobs finish (hours).
  - `touch /home/exouser/henry/Qwen_Drive/state/STOP`  — we stop all our GPU jobs within ~1 min (we lose <= 1 epoch).
  - `rm -f /home/exouser/henry/Qwen_Drive/state/STOP /home/exouser/henry/Qwen_Drive/state/PAUSE` — we resume by ourselves within 1 min.
  - You can leave a line in `/home/exouser/henry/Qwen_Drive/state/NOTE` (e.g. "need the whole GPU Fri 18:00-22:00 UTC").
- Please do not `kill`/`pkill python`, `nvidia-smi -r`, reboot, upgrade the NVIDIA driver / CUDA / docker, or
  `apt autoremove` without telling Henry; if you must reboot, `touch STOP` first and wait 2 minutes.
- Disk: we stay under ~500 GB of the 1.3 TB root disk; please keep >= 100 GB free.
- Our crontab lines (`crontab -l`) restart the queue after a reboot and refresh a status file every 10 min; please leave them.
- Contact Henry: <HENRY_CONTACT>

**中文**
- Henry 的实验在这个账号下运行，tmux 会话名 `fdq`（`tmux ls`；只看不动：`tmux attach -r -t fdq`）。
  我们的全部文件在 `/home/exouser/henry/`（另有符号链接 `/home/henry/Desktop/Qwen_Drive` 指向其中），请不要删除、移动或修改。服务器时间为 UTC。
- 显存：Henry 的任务在这张卡上优先，可能用满全部显存（Henry 2026-10-18 决定）。你需要用 GPU 时请用下面的标记文件
  （PAUSE / STOP）或留 NOTE，我们一分钟内让出；绝不会 kill 你的进程。
  我们的进程是工作目录在 `/home/exouser/henry` 下的 `python`（`ls -l /proc/<pid>/cwd` 可查），绝不会动你的进程。
- 需要更多显存 / CPU / 内存时请用标记文件，不必 kill：
  - `touch /home/exouser/henry/Qwen_Drive/state/PAUSE` —— 我们不再启动新任务，正在跑的跑完（几小时）
  - `touch /home/exouser/henry/Qwen_Drive/state/STOP` —— 我们约 1 分钟内停掉全部 GPU 任务（最多损失 1 个 epoch）
  - `rm -f /home/exouser/henry/Qwen_Drive/state/STOP /home/exouser/henry/Qwen_Drive/state/PAUSE` —— 1 分钟内自动恢复
  - 可以在 `/home/exouser/henry/Qwen_Drive/state/NOTE` 留言（如"周五 18:00–22:00 UTC 需要整卡"）。
- 请勿在未告知 Henry 的情况下 kill 我们的进程、`nvidia-smi -r`、重启、升级驱动 / CUDA / docker 或 `apt autoremove`；
  必须重启时请先 `touch STOP` 并等 2 分钟。
- 磁盘：我们占用不超过约 500 GB（根盘 1.3 TB），请保留 ≥ 100 GB 空闲。
- crontab 里我们的几行用于重启后恢复队列和每 10 分钟刷新状态文件，请保留。
- 联系 Henry：<HENRY_CONTACT>
