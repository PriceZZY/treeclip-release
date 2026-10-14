#!/usr/bin/env bash
# Phase-3b: cited, view-dependent generators not in run_phase3.sh.
# source_control (self-loops both datasets), rev_predict_source (per-dataset),
# and the two reconstructed generators (supervised_matched_sf, rev_perclass_for).
cd "$(dirname "$0")/../.." || exit 1
run () { echo ">>> $*"; if uv run python "scripts/protocol_v2/$1.py" "${@:2}"; then echo "OK $*"; else echo "FAIL $*"; fi; }
run gen_source_control
run rev_predict_source for
run rev_predict_source biodiv
run gen_supervised_matched_sf
run gen_rev_perclass
echo "PHASE3B_CHAIN_DONE"
