#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CIRCUITNET_ROOT="${CIRCUITNET_ROOT:-${HOME}/datasets/CircuitNet/CircuitNet-N14}"
RUN_ROOT="${RUN_ROOT:-runs/real_tier1}"
PROVIDER="${PROVIDER:-mock}"
POPULATION_SIZE="${POPULATION_SIZE:-10}"
GENERATIONS="${GENERATIONS:-2}"
ELITE_COUNT="${ELITE_COUNT:-5}"

cd "$ROOT_DIR"
export CIRCUITNET_ROOT

python3 -m pytest
python3 -m coevop.cli env-check

mkdir -p data_manifests "$RUN_ROOT"

python3 -m coevop.cli circuitnet-manifest \
  --root "$CIRCUITNET_ROOT" \
  --output data_manifests/circuitnet_n14_real.local.json

for design in Vortex-small nvdla-small openc910-1; do
  python3 -m coevop.cli circuitnet-manifest \
    --root "$CIRCUITNET_ROOT" \
    --design "$design" \
    --output "data_manifests/${design}.local.json"
done

python3 -m coevop.cli real-tier1-audit \
  --manifest data_manifests/circuitnet_n14_real.local.json \
  --output-dir "$RUN_ROOT/baseline" \
  --split-strategy auto

python3 -m coevop.cli evolve-offline \
  --manifest data_manifests/circuitnet_n14_real.local.json \
  --provider mock \
  --run-dir "$RUN_ROOT/mock_full" \
  --population-size 10 \
  --generations 2 \
  --elite-count 5 \
  --split-strategy auto \
  --selection-split validation \
  --reflection-mode accepted \
  --islands 5 \
  --enable-constant-fit

python3 -m coevop.cli real-tier1-audit \
  --manifest data_manifests/circuitnet_n14_real.local.json \
  --output-dir "$RUN_ROOT/mock_full" \
  --split-strategy auto \
  --evolution-dir "$RUN_ROOT/mock_full"

for validation_design in Vortex-small nvdla-small openc910-1; do
  panel_dir="$RUN_ROOT/mock_lodo_${validation_design}"
  python3 -m coevop.cli evolve-offline \
    --manifest data_manifests/circuitnet_n14_real.local.json \
    --provider mock \
    --run-dir "$panel_dir" \
    --population-size 6 \
    --generations 1 \
    --elite-count 3 \
    --selection-split validation \
    --validation-design "$validation_design" \
    --reflection-mode accepted \
    --islands 5 \
    --enable-constant-fit
  python3 -m coevop.cli real-tier1-audit \
    --manifest data_manifests/circuitnet_n14_real.local.json \
    --output-dir "$panel_dir" \
    --validation-design "$validation_design" \
    --evolution-dir "$panel_dir"
done

if [[ "$PROVIDER" == "qwen" ]]; then
  if [[ -z "${QWEN_API_KEY:-}${DASHSCOPE_API_KEY:-}" ]]; then
    echo "PROVIDER=qwen requested, but QWEN_API_KEY/DASHSCOPE_API_KEY is not set." >&2
    exit 2
  fi
  python3 -m coevop.cli evolve-offline \
    --manifest data_manifests/circuitnet_n14_real.local.json \
    --provider qwen \
    --run-dir "$RUN_ROOT/qwen_full_v06" \
    --population-size 50 \
    --generations 3 \
    --elite-count 10 \
    --split-strategy auto \
    --selection-split validation \
    --reflection-mode accepted \
    --operator-weights crossover=0.35,mutate=0.30,param_tune=0.20,simplify=0.15 \
    --islands 5 \
    --enable-constant-fit
  python3 -m coevop.cli real-tier1-audit \
    --manifest data_manifests/circuitnet_n14_real.local.json \
    --output-dir "$RUN_ROOT/qwen_full_v06" \
    --split-strategy auto \
    --evolution-dir "$RUN_ROOT/qwen_full_v06"
fi

echo "Real Tier-1 run complete under $ROOT_DIR/$RUN_ROOT"
