#!/usr/bin/env bash
# Phase-3: re-run every eval that consumes the re-extracted (n=6) feature files.
# Each step prints "OK <name>" or "FAIL <name>" so progress/failures are traceable.
# gen_unsup_robust_multi already ran (gate 2) but is included for a complete, idempotent sweep.
cd "$(dirname "$0")/../.." || exit 1
run () {
  echo ">>> $*"
  if uv run python "scripts/protocol_v2/$1.py" "${@:2}"; then echo "OK $*"; else echo "FAIL $*"; fi
}
run gen_unsup_robust_multi for
run gen_unsup_robust_multi biodiv
run gen_unsup_robust for
run gen_unsup_robust biodiv
run gen_source_control_multi for
run gen_source_control_multi biodiv
run gen_source_control_robust
run gen_source_control_robust_dino
run gen_permutation_multi for
run gen_permutation_multi biodiv
run gen_permutation_test
run gen_permutation_test_dino
run gen_experiment_battery_dino for
run gen_experiment_battery_dino biodiv
run gen_experiment_battery for
run gen_experiment_battery biodiv
run gen_score for
run gen_score biodiv
run gen_qualitative
run gen_qualitative_dino
echo "PHASE3_CHAIN_DONE"
