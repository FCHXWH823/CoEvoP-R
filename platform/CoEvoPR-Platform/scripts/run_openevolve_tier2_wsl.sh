#!/usr/bin/env bash
set -euo pipefail

# Run the current OpenEvolve-style direct Tier-2 DREAMPlace evolution flow.
#
# Intended use inside WSL/Linux from the CoEvoPR-Platform repository root:
#
#   export OPENAI_API_KEY=...
#   ./scripts/run_openevolve_tier2_wsl.sh smoke
#
# Modes:
#   smoke       one-design/short GPT-4.1-mini smoke run
#   multismoke  multi-design short GPT-4.1-mini smoke run
#   date        DATE-level direct Tier-2 config; expensive
#
# The script deliberately does not read API-key files and does not set any key.

MODE="${1:-smoke}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DREAMPLACE_ROOT="${DREAMPLACE_ROOT:-${HOME}/DREAMPlace/install}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4.1-mini}"

case "$MODE" in
  smoke)
    CONFIG="configs/openevolve_tier2/chipbench_smoke_gpt41mini.toml"
    RUN_DIR="runs/openevolve_tier2/chipbench_smoke_gpt41mini_$(date +%Y%m%d_%H%M%S)"
    ;;
  multismoke)
    CONFIG="configs/openevolve_tier2/chipbench_direct_multidesign_smoke_gpt41mini.toml"
    RUN_DIR="runs/openevolve_tier2/chipbench_direct_multidesign_smoke_gpt41mini_$(date +%Y%m%d_%H%M%S)"
    ;;
  date)
    CONFIG="configs/openevolve_tier2/chipbench_direct_tier2.toml"
    RUN_DIR="runs/openevolve_tier2/chipbench_direct_tier2_$(date +%Y%m%d_%H%M%S)"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 [smoke|multismoke|date]" >&2
    exit 2
    ;;
esac

echo "[coevop] repo: $REPO_ROOT"
echo "[coevop] config: $CONFIG"
echo "[coevop] run_dir: $RUN_DIR"
echo "[coevop] DREAMPLACE_ROOT: $DREAMPLACE_ROOT"
echo "[coevop] OPENAI_MODEL: $OPENAI_MODEL"

python3 -m pytest tests/test_openevolve_core.py tests/test_openevolve_tier2.py tests/test_openevolve_configs.py -p no:cacheprovider
python3 -m coevop.cli llm-check
python3 -m coevop.cli dreamplace-status

python3 -m coevop.cli openevolve-tier2 \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --resume

python3 -m coevop.cli openevolve-prompt-audit \
  --run-dir "$RUN_DIR" \
  --output "$RUN_DIR/prompt_audit_summary.json"

echo "[coevop] completed: $RUN_DIR"
echo "[coevop] summarize with:"
echo "  python3 scripts/summarize_openevolve_run.py $RUN_DIR"
