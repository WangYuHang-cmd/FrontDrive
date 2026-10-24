# 实验台账（ledger）字段规范（C-15；工具待 R-03 决定，CSV 为工具无关的最小集）

每个训练 / 评估 run 一行，文件 `frontdrive/eval/ledger.csv`（UTF-8，逗号分隔）。四种"步数"必须分列（指南 §9 第 5 条）。

| 列 | 含义 |
|---|---|
| run_id | `YYYYMMDD-HHMM_<group>_<seed>` |
| date | 运行日期 |
| group | 实验组编号（计划 §7A：1 / 1b / 1c / 2 / 3 / 4 / 5 / 6 / 7 / L） |
| fusion_variant | none / pooled / xattn / input_concat / dummy_const / dummy_qwenpool |
| T | memory token 数（32 / 64 / 128；无则空） |
| branch_layers | 如 `0,4,8,12,16,20,24,28` |
| compressor | perceiver / pool / none |
| probe_version | 探针脚本版本（F1 用） |
| teacher_ckpt | BADAS / fcamera 教师 checkpoint 标识（无则空） |
| planner_ckpt | planner-sft 的 sha256 前 8 位 |
| data_subset | 训练子集 manifest 标识（如 `pilot100_v1`、`sub300h_v1`） |
| hours | 训练子集小时数 |
| microsteps | 前向反向次数 |
| optimizer_steps | 优化器更新次数 |
| draws | 每场景 draw 数 × 场景数（总 (t, τ₀) 采样数） |
| distinct_scenes | 独立场景覆盖量 |
| seed | 随机种子 |
| trainable_params_M | 可训练参数量 |
| gpu_hours | 实际 GPU 小时（含评估） |
| peak_vram_gb | 峰值显存 |
| eval_set | comma_valdev / wod_valdev / wod_valholdout / navtest / … |
| is_oracle | best-of-K 选择是否用了真值（Y/N）；榜单只收 N |
| num_samples | K |
| ade3 / ade5 / fde5 / rfs / pdms / ep | 指标（空则未测） |
| ci_low / ci_high | 配对 bootstrap 95% CI（相对基线） |
| evidence_path | `docs/evidence/<日期>_<主题>/` |
| notes | 自由文本（不含凭据 / PII） |

## v4 新增列（2026-10-11）

| 列 | 含义 |
|---|---|
| encoder_ckpt_sha8 | 编码器权重 sha256 前 8 位 |
| encoder_init | vitl / e50 / comma_from_e50 / comma_from_vitl / … |
| encoder_frozen | Y/N（冻结只能用于诊断） |
| protocol | pf（perception-free）/ pb（perception-based） |
| pretrain_subset | manifest id（如 subset_v0） |
| pretrain_objective | jepa / jepa+motion / … |
| pretrain_steps / pretrain_tokens / pretrain_flops_est | 续训账目（加头后"相同步数 ≠ 相同算力"） |
| clip_draws | 续训抽样的 clip 次数 |
| n_scenes | 评估场景数 |
| cluster_ci_low / cluster_ci_high | 按 log 聚类的配对 bootstrap 95% CI（相对基线） |
| step_time_s | 训练步耗时 |
| dev_or_test | dev（navtrain 留出）/ navtest；navtest 每变体 ≤2 次 |
