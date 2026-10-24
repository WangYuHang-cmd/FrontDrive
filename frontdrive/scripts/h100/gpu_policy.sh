# GPU sharing policy for every launcher on the H100 (sourced by phase5_pb_pipeline.sh, eval_helper.sh, pb_queue.sh,
# phase6_pf_fillers.sh). Henry 2026-10-18: "我方显存不需要设置限制, 我的优先级是最高的; 队友仅会在极少情况下使用, 并且如果我在用他们会用其他的"
# -> no self-imposed cap and no reservation for the other user (the earlier 60 GB / 20 GB rule of plan H.10 ① is withdrawn);
# only a small margin against our own OOM. The other user still has the STOP / PAUSE / NOTE markers (README_SHARED_GPU.md).
# Every value can be overridden through the environment; the queue re-reads this file on each poll, so edits apply live.
TOTAL_MIB=81559
CAP_MIB=${CAP_MIB:-81559}                 # our total allowance (MiB); 81559 = the whole card
FRIEND_RESERVE_MIB=${FRIEND_RESERVE_MIB:-0}   # MiB kept free for the other user; 0 per Henry
OOM_MARGIN_MIB=${OOM_MARGIN_MIB:-2000}    # never fill the last 2 GB (allocator fragmentation, CUDA context growth)
# NOTE (2026-10-22 09:25 EDT): `X=${X:-default}` is NOT re-read by a long-running shell that already sourced this file once (the
# variable is set, so the default is ignored) -> pb_queue.sh kept MAX_PB=2 after the file said 3. The live-tunable values therefore
# take their defaults from *_OVERRIDE environment names: editing the number below and rsync-ing the file changes the running queue
# at its next poll; set MAX_PB_OVERRIDE / LOAD_MAX_LAUNCH_OVERRIDE in the environment only to pin a value for one launcher.
MAX_PB=${MAX_PB_OVERRIDE:-2}              # pb training chains at once. 2 while a V-JEPA pretraining arm (A4, CPU-bound decode) runs beside
                                          # them: 2026-10-22 09:35 EDT measured 3 pb + arm = load1 41, 7.7 s/step each (summed throughput
                                          # -16 % vs 2 pb + arm at 4.3 s/step) and the eval dispatcher blocked (LOAD_MAX 40). Was 3 for
                                          # 20 min to get the ⑤ probe started. Back to 3 when no arm runs (2026-10-19: 3 pb + pf + 2 evals = 57).
LOAD_MAX=${LOAD_MAX:-40}                  # load1 above which no new eval is dispatched (20 vCPU host)
# 2026-10-19 00:05 EDT observation: 3 pb trainers + a pf trainer + 2 eval streams pushed load1 to 57-59 and the pb step times
# to 5.9-7.7 s (vs 2.6-3.5 with two trainers): the host thrashes and the summed throughput FALLS. New *training* launches
# therefore also require load1 <= LOAD_MAX_LAUNCH; evals keep LOAD_MAX (they are what the queued-up exports wait for).
LOAD_MAX_LAUNCH=${LOAD_MAX_LAUNCH_OVERRIDE:-34}   # 34 from 2026-10-22 09:25 EDT (was 30): with 3 trainers + the A4 arm the host idles at load 27-30,
                                                #  so the ⑤ probe launch must not hang on a 31; once it runs, load > 34 blocks further launches as intended
free_for_us() {  # <need MiB> <gpu used MiB> <ours MiB> -> 0 when a job needing <need> fits under the policy
  local need=$1 used=$2 ours=$3
  [ $((ours + need)) -le "$CAP_MIB" ] && [ $((TOTAL_MIB - used - need)) -ge $((FRIEND_RESERVE_MIB + OOM_MARGIN_MIB)) ]
}
