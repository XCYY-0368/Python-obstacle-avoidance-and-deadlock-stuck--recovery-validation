#!/usr/bin/env bash
#
# run_tests.sh -- convenience wrapper around run_tests.py for the container sim.
#
# Subcommands:
#   ./run_tests.sh                 full default sweep (12 seeds, ~5 min)
#   ./run_tests.sh quick           2-seed smoke test (~45 s)
#   ./run_tests.sh sweep 1 2 3     sweep specific seeds
#   ./run_tests.sh golden          run the default sweep AND save it as the golden baseline
#   ./run_tests.sh viz 12          baseline-vs-COORD trajectory PNG for one seed
#   ./run_tests.sh anim 12         COORD animation GIF for one seed (slow)
#   ./run_tests.sh setup           create .venv and install numpy/scipy/matplotlib/pillow
#
# All Python heavy lifting lives in run_tests.py / viz.py; this is just ergonomics.

set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}

cmd=${1:-sweep}

case "$cmd" in
  setup)
    $PY -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip
    pip install numpy scipy matplotlib pillow
    echo "Done. Activate with:  source .venv/bin/activate"
    ;;

  quick)
    $PY run_tests.py --seeds 12 88 --determinism-seeds 88
    ;;

  sweep)
    shift || true
    if [ "$#" -gt 0 ]; then
      $PY run_tests.py --seeds "$@"
    else
      $PY run_tests.py
    fi
    ;;

  golden)
    $PY run_tests.py --update-golden
    ;;

  viz)
    seed=${2:?usage: ./run_tests.sh viz <seed>}
    $PY - "$seed" <<'PYEOF'
import sys, viz
seed = int(sys.argv[1])
viz.coord_compare([seed], fname="coord_compare_seed%s.png" % seed)
PYEOF
    echo "see outputs/coord_compare_seed${seed}.png"
    ;;

  anim)
    seed=${2:?usage: ./run_tests.sh anim <seed>}
    $PY - "$seed" <<'PYEOF'
import sys, viz
seed = int(sys.argv[1])
viz.coord_animation(seed, fname="anim_coord_seed%s.gif" % seed)
PYEOF
    echo "see outputs/anim_coord_seed${seed}.gif"
    ;;

  *)
    echo "unknown subcommand: $cmd"
    echo "try: setup | quick | sweep [seeds...] | golden | viz <seed> | anim <seed>"
    exit 2
    ;;
esac
