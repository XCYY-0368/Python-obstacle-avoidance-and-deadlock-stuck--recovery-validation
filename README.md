# Container Corridor Multi-Robot Benchmark

A lightweight pure-Python benchmark of three differential-drive robots running an
endless pick-and-place loop on a long, narrow ("container") map. Layered architecture:

- **Global planner**: A* polyline path + line-of-sight simplification
- **Local avoidance**: ORCA (velocity-obstacle), distributed via a shared bulletin board
- **Optional coordination layer** (`coordination.py`): per-bottleneck priority mutex +
  dynamic jam token rotation, sitting on top of ORCA

## Run

```bash
# baseline (ORCA only)
SEED=12 MAX_STEPS=5000 STEER_MODE=arch1 python container_sim.py

# with coordination layer
COORD=1 SEED=12 MAX_STEPS=5000 STEER_MODE=arch1 python container_sim.py
```

Environment variables: `SEED`, `MAX_STEPS`, `COORD` (0/1), `STEER_MODE` (arch1/arch2).

## Files

- `container_sim.py` — simulation, map generation, annotation, planner, ORCA, follower
- `coordination.py` — bottleneck + jam coordination layer
- `CODE_REVIEW.md` — systematic review, fixes, and seed-12 iteration log

## Dependencies

`numpy`, `scipy` (for map annotation connectivity analysis).

## Branches

- `main` — stable baseline
- `develop` — integration branch
- `feature/*` — individual improvements (see git log)
