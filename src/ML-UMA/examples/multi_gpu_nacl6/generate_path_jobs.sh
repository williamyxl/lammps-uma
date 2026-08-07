#!/usr/bin/env bash
# Generate one-path-per-job SLURM scripts (VRAM isolation policy).
# Usage: ./generate_path_jobs.sh
set -euo pipefail

EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="${EX}/_run_common.sh"
GP_COMMON="${EX}/gp_round/_run_gp_common.sh"

write_suite() {
  local ngpus="$1" path="$2" kind="$3"  # kind=suite|gp
  local gpus="$ngpus"
  local cpus mem time job name out
  case "$ngpus" in
    1) cpus=16; mem=56G; time=02:00:00 ;;
    2) cpus=32; mem=112G; time=02:00:00 ;;
    4) cpus=64; mem=224G; time=02:00:00 ;;
    *) echo "bad ngpus $ngpus" >&2; exit 1 ;;
  esac

  local short
  case "$path" in
    ase) short=ase ;;
    fc) short=fc ;;
    uma_double) short=d64 ;;
    uma_mixed) short=mix ;;
    *) echo "bad path $path" >&2; exit 1 ;;
  esac

  if [[ "$kind" == "gp" ]]; then
    name="uma-gp-n${ngpus}-${short}"
    out="${EX}/gp_round/run_ngpu${ngpus}_${path}.slurm"
    cat >"$out" <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${gpus}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=gp_round/%x-%j.out
#SBATCH --error=gp_round/%x-%j.out

# VRAM isolation: ONE path per job (${path} @ ${ngpus} GPU)
set -euo pipefail
export NGPUS=${ngpus}
export UMA_DEVICES=${ngpus}
export ONLY_PATHS=${path}
export RECOMPILE="\${RECOMPILE:-0}"
export MERGE_RESULTS=1
export SKIP_DIST_DESTROY=1
source ${GP_COMMON}
EOF
  else
    name="uma-mgpu-n${ngpus}-${short}"
    out="${EX}/run_ngpu${ngpus}_${path}.slurm"
    cat >"$out" <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --account=bbpl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${gpus}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# VRAM isolation: ONE path per job (${path} @ ${ngpus} GPU)
set -euo pipefail
export NGPUS=${ngpus}
export UMA_DEVICES=${ngpus}
export ONLY_PATHS=${path}
export FAIRCHEM_WORKERS=${ngpus}
export RECOMPILE="\${RECOMPILE:-0}"
export MERGE_RESULTS=1
export SKIP_DIST_DESTROY=1
EOF
    if [[ "$path" == "fc" ]]; then
      cat >>"$out" <<'EOF'
export HARD_EXIT_AFTER_FC=1
EOF
    fi
    cat >>"$out" <<EOF
source ${COMMON}
EOF
  fi
  chmod +x "$out"
  echo "wrote $out"
}

for n in 1 2 4; do
  for p in ase fc uma_double uma_mixed; do
    write_suite "$n" "$p" suite
  done
  for p in uma_double uma_mixed; do
    write_suite "$n" "$p" gp
  done
done

echo "Done. Prefer: ./submit_path_jobs.sh"
