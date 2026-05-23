#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

GYM_ENV="${GYM_ENV:-legged_gym}"
LAB_ENV="${LAB_ENV:-/home/cra02/anaconda3/envs/env_isaaclab}"
GYM_SESSION="${GYM_SESSION:-hgex_rough_gym_1000}"
LAB_SESSION="${LAB_SESSION:-hgex_rough_lab_1000}"

GYM_ROUGH_LOG="${GYM_ROUGH_LOG:-/tmp/hgex_rough_heights_curric_gym_4096_seed42_1000.log}"
LAB_ROUGH_LOG="${LAB_ROUGH_LOG:-/tmp/hgex_rough_heights_curric_lab_critic_only_4096_seed42_1000.log}"
OUT_DIR="${OUT_DIR:-/tmp/hgex_original_scene_matrix}"
mkdir -p "${OUT_DIR}"

wait_for_session() {
  local session="$1"
  while tmux has-session -t "${session}" 2>/dev/null; do
    echo "[matrix] waiting for tmux session ${session}"
    sleep 300
  done
}

latest_run_dir() {
  local root="$1"
  local suffix="$2"
  find "${root}" -maxdepth 1 -type d -name "*${suffix}" | sort | tail -1
}

run_pair() {
  local gym_cmd="$1"
  local lab_cmd="$2"
  bash -lc "${gym_cmd}" &
  local gym_pid=$!
  bash -lc "${lab_cmd}" &
  local lab_pid=$!
  wait "${gym_pid}"
  wait "${lab_pid}"
}

echo "[matrix] waiting for rough long training runs"
wait_for_session "${GYM_SESSION}"
wait_for_session "${LAB_SESSION}"

echo "[matrix] comparing rough long training curves"
python humanoid_gym_ex/scripts/compare_training_curves.py \
  --isaacgym-log "${GYM_ROUGH_LOG}" \
  --isaaclab-log "${LAB_ROUGH_LOG}" \
  --tail 10 \
  --csv "${OUT_DIR}/rough_heights_curric_curve.csv" \
  | tee "${OUT_DIR}/rough_heights_curric_curve.txt"

GYM_ROUGH_DIR="$(latest_run_dir logs/XBot_ppo rough_heights_curric_gym_4096_seed42_1000)"
LAB_ROUGH_DIR="$(latest_run_dir logs/XBot_isaaclab rough_heights_curric_lab_critic_only_4096_seed42_1000)"
GYM_ROUGH_CKPT="${GYM_ROUGH_DIR}/model_1000.pt"
LAB_ROUGH_CKPT="${LAB_ROUGH_DIR}/model_1000.pt"

if [[ ! -f "${GYM_ROUGH_CKPT}" || ! -f "${LAB_ROUGH_CKPT}" ]]; then
  echo "[matrix] missing rough checkpoints: ${GYM_ROUGH_CKPT} ${LAB_ROUGH_CKPT}" >&2
  exit 1
fi

echo "[matrix] running rough final-checkpoint native replay"
run_pair \
  "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${ROOT_DIR} HGEX_DIAG_STEPS=600 HGEX_DIAG_FIX_COMMAND=1 HGEX_DIAG_CHECKPOINT=${GYM_ROUGH_CKPT} HGEX_DIAG_OUTPUT=${OUT_DIR}/rough_gym_policy_gym_backend.json HGEX_DIAG_TRACE_OUTPUT=${OUT_DIR}/rough_gym_policy_gym_backend_trace.json conda run --no-capture-output -n ${GYM_ENV} python humanoid_gym_ex/scripts/diagnose_isaacgym_rollout.py --task=humanoid_ppo --headless --num_envs 64 --terrain rough --measure_heights --terrain_curriculum --sim_device cuda:0 --rl_device cuda:0 2>&1 | tee ${OUT_DIR}/rough_gym_policy_gym_backend.log" \
  "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${ROOT_DIR} conda run --no-capture-output -p ${LAB_ENV} python humanoid_gym_ex/scripts/diagnose_isaaclab_rollout.py --headless --num_envs 64 --steps 600 --fix_command --terrain rough --measure_heights --terrain_curriculum --checkpoint_path ${LAB_ROUGH_CKPT} --output ${OUT_DIR}/rough_lab_policy_lab_backend.json --trace_output ${OUT_DIR}/rough_lab_policy_lab_backend_trace.json --device cuda:0 2>&1 | tee ${OUT_DIR}/rough_lab_policy_lab_backend.log"

echo "[matrix] running rough final-checkpoint cross-backend replay"
run_pair \
  "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${ROOT_DIR} HGEX_DIAG_STEPS=600 HGEX_DIAG_FIX_COMMAND=1 HGEX_DIAG_CHECKPOINT=${LAB_ROUGH_CKPT} HGEX_DIAG_OUTPUT=${OUT_DIR}/rough_lab_policy_gym_backend.json HGEX_DIAG_TRACE_OUTPUT=${OUT_DIR}/rough_lab_policy_gym_backend_trace.json conda run --no-capture-output -n ${GYM_ENV} python humanoid_gym_ex/scripts/diagnose_isaacgym_rollout.py --task=humanoid_ppo --headless --num_envs 64 --terrain rough --measure_heights --terrain_curriculum --sim_device cuda:0 --rl_device cuda:0 2>&1 | tee ${OUT_DIR}/rough_lab_policy_gym_backend.log" \
  "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${ROOT_DIR} conda run --no-capture-output -p ${LAB_ENV} python humanoid_gym_ex/scripts/diagnose_isaaclab_rollout.py --headless --num_envs 64 --steps 600 --fix_command --terrain rough --measure_heights --terrain_curriculum --checkpoint_path ${GYM_ROUGH_CKPT} --output ${OUT_DIR}/rough_gym_policy_lab_backend.json --trace_output ${OUT_DIR}/rough_gym_policy_lab_backend_trace.json --device cuda:0 2>&1 | tee ${OUT_DIR}/rough_gym_policy_lab_backend.log"

echo "[matrix] running heightfield/trimesh parser and train-entry smoke"
for terrain in heightfield trimesh; do
  run_pair \
    "WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${ROOT_DIR} conda run --no-capture-output -n ${GYM_ENV} python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1 --seed 42 --run_name alias_${terrain}_gym_seed42_1 --terrain ${terrain} --measure_heights --terrain_curriculum --sim_device cuda:0 --rl_device cuda:0 2>&1 | tee ${OUT_DIR}/${terrain}_gym_smoke.log" \
    "WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${ROOT_DIR} conda run --no-capture-output -p ${LAB_ENV} python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 64 --num_steps_per_env 60 --max_iterations 1 --seed 42 --run_name alias_${terrain}_lab_seed42_1 --terrain ${terrain} --measure_heights --terrain_curriculum --device cuda:0 2>&1 | tee ${OUT_DIR}/${terrain}_lab_smoke.log"
done

echo "[matrix] running full smoke validation"
bash humanoid_gym_ex/scripts/validate_smoke.sh 2>&1 | tee "${OUT_DIR}/validate_smoke.log"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT_DIR}/DONE"
echo "[matrix] done: ${OUT_DIR}"
