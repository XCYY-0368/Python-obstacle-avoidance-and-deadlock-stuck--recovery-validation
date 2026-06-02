"""
container_sim.py
================
A lightweight, pure-Python multi-robot benchmark on a long, narrow ("container")
map. Three differential-drive robots run an endless pick-and-place loop: each
robot is given a random goal, drives there using ORCA for local avoidance, then
gets a new random goal, forever. We measure how long the system runs without a
collision and without a global deadlock.

Design follows the project handoff conventions:
  - Differential-drive (nonholonomic) robot, state (x, y, theta), control (v, w).
  - R_min = 1.2 m, v_max = 1.0 m/s, physical radius 0.5 m, SAFETY_FACTOR = 2.0.
  - Distributed in spirit: each robot only reads a shared "bulletin board" of
    neighbor states; no central planner drives the avoidance.
  - Environment switches via env vars: SEED, MAX_STEPS, ALGO (orca only here).

MAP-ANNOTATION STAGE (offline, before robots start)
---------------------------------------------------
After the random map is generated we analyze it:
  1. Rasterize free space at the inflated robot radius.
  2. Find narrow gaps (corridor width < 2 * inflation_radius = 2.96 m).
  3. For each narrow gap, temporarily block it and test connectivity:
       - still connected  -> "reroutable narrow gap": NO token, solved by
         priority + replanning.
       - disconnected     -> "coordination bottleneck" (a cut / unique path):
         place a TOKEN and two WAIT ZONES, one on each side.
In a 5 m x 15 m open corridor with small scattered obstacles, true unique-path
cut points are very unlikely to form, so we expect zero coordination bottlenecks
and the run degrades to pure ORCA + priority. That outcome is itself recorded.
"""

import os
import math
import numpy as np
from collections import deque
import coordination as coord_mod

COORD = os.environ.get("COORD", "0") == "1"   # enable bottleneck coordination layer

# ---------------------------------------------------------------------------
# Global physical / algorithm parameters (per handoff)
# ---------------------------------------------------------------------------
ROBOT_RADIUS   = 0.5          # physical radius [m]
SAFETY_FACTOR  = 1.5
V_MAX          = 1.0          # max linear speed [m/s]
R_MIN          = 1.2          # min turning radius [m]
W_MAX          = V_MAX / R_MIN  # max angular speed bound by R_min while DRIVING
W_TURN         = 2.0 * W_MAX    # in-place turning (v=0) is NOT bound by R_min
DT             = 0.1          # time step [s]
INFLATION_R    = ROBOT_RADIUS * (1.0 + 0.48)  # ~0.74; planning radius ~ matches handoff note
# Effective avoidance radius used by ORCA for OTHER ROBOTS (dynamic): physical * safety
ORCA_RADIUS    = ROBOT_RADIUS * SAFETY_FACTOR  # 1.0 m
# Static walls/obstacles do NOT move, so they need only a small physical margin,
# NOT the dynamic safety factor. Robots may pass close to walls safely.
STATIC_CLEAR   = ROBOT_RADIUS + 0.15           # 0.65 m clearance from static surfaces

# Map dimensions: 5 robot-widths x 15 robot-widths (robot diameter = 1 m)
MAP_W          = 5.0          # short side (x), [m]
MAP_H          = 15.0         # long side (y), [m]

GOAL_TOL       = 0.4          # reached-goal tolerance [m]
NEIGHBOR_HORIZON = 4.0        # ORCA time horizon (tau) seconds
DEADLOCK_PATIENCE = 250       # steps of ~no progress before a robot is "locked"
PROGRESS_EPS   = 0.05         # min displacement over patience window to count as progress
RECOVER_SLEEP  = 100          # steps a soft-locked robot sleeps before retrying its goal (#12)
                              # 10 s at dt=0.1: longer sleep gives the coordination
                              # layer (token holder) more room to route around the
                              # frozen robot before it wakes and rejoins. (#15)
HARD_LOCK_ATTEMPTS = 8        # after this many failed revivals, treat as a genuine hard lock (#12)
STALL_TIMEOUT  = 300          # steps without getting closer to the goal -> deadlock (#14)
PROGRESS_DELTA = 0.15         # min distance reduction toward goal that counts as progress (#14)

INFLATION_FACTOR_FOR_GAP = 1.48  # used to define bottleneck width threshold
GAP_WIDTH_THRESHOLD = 2.0 * INFLATION_FACTOR_FOR_GAP  # 2.96 m


# ---------------------------------------------------------------------------
# Obstacles & map
# ---------------------------------------------------------------------------
class Disc:
    """Static circular obstacle."""
    __slots__ = ("x", "y", "r")
    def __init__(self, x, y, r):
        self.x = x; self.y = y; self.r = r


def make_container_map(rng, n_obstacles=3):
    """Generate the container map: walls (as discs) + random fixed obstacles.

    Obstacles are fixed at ONE robot-size (radius = ROBOT_RADIUS = 0.5 m), exactly
    3 of them, randomly placed but: (a) never inside the initial zone, (b) spaced
    apart so the corridor is not sealed, (c) off the walls. Placement also
    guarantees a single-robot-passable route remains (checked by caller via
    annotation; here we just keep clearance reasonable).
    """
    bounds = (0.0, MAP_W, 0.0, MAP_H)  # xmin, xmax, ymin, ymax

    # Walls discretized into overlapping static discs along the rectangle border.
    wall_r = 0.25
    walls = []
    step = wall_r * 1.5
    x = 0.0
    while x <= MAP_W:
        walls.append(Disc(x, 0.0, wall_r))
        walls.append(Disc(x, MAP_H, wall_r))
        x += step
    y = 0.0
    while y <= MAP_H:
        walls.append(Disc(0.0, y, wall_r))
        walls.append(Disc(MAP_W, y, wall_r))
        y += step

    # Random fixed obstacles: exactly one-robot-size.
    OBST_R = ROBOT_RADIUS              # one robot-size (radius 0.5 m)
    INIT_ZONE = (0.5, 4.5, 0.5, 7.5)   # x0,x1,y0,y1 keep-clear initial zone
    obstacles = []
    for spacing in (0.8, 0.4, 0.0):    # relax spacing if needed to fit 3
        attempts = 0
        while len(obstacles) < n_obstacles and attempts < 5000:
            attempts += 1
            ox = rng.uniform(STATIC_CLEAR + OBST_R, MAP_W - STATIC_CLEAR - OBST_R)
            oy = rng.uniform(8.0, MAP_H - 1.0)   # corridor body, above the initial zone
            if INIT_ZONE[0] <= ox <= INIT_ZONE[1] and INIT_ZONE[2] <= oy <= INIT_ZONE[3]:
                continue
            ok = all(math.hypot(ox-o.x, oy-o.y) >= (OBST_R + o.r + 2*ROBOT_RADIUS + spacing)
                     for o in obstacles)
            if ok:
                obstacles.append(Disc(ox, oy, OBST_R))
        if len(obstacles) >= n_obstacles:
            break
    return walls, obstacles, bounds


# ---------------------------------------------------------------------------
# MAP ANNOTATION: bottleneck detection (narrow AND unique-path)
# ---------------------------------------------------------------------------
def annotate_map(obstacles, bounds, cell=0.1):
    """Offline map annotation. Identifies bottlenecks that need COORDINATION,
    using the user's definition: a passage is a coordination bottleneck iff it is
    BOTH (a) narrow (corridor width < GAP_WIDTH_THRESHOLD) AND (b) the UNIQUE path
    between the two ends (removing it disconnects bottom from top). Narrow gaps
    that can be bypassed (still connected after removal) are NOT coordination
    bottlenecks; they are left to priority + replanning.

    Returns dict with: reachable mask, list of bottlenecks (each with a position
    and two wait-zone points, one per side), reroutable narrow gaps, width profile.
    """
    from scipy import ndimage
    xmin, xmax, ymin, ymax = bounds
    nx = int(round((xmax - xmin) / cell)) + 1
    ny = int(round((ymax - ymin) / cell)) + 1
    xs = xmin + np.arange(nx) * cell
    ys = ymin + np.arange(ny) * cell
    XX, YY = np.meshgrid(xs, ys, indexing="xy")  # (ny, nx)

    # Clearance to nearest static surface (walls = rectangle border; obstacles).
    border = np.minimum.reduce([XX - xmin, xmax - XX, YY - ymin, ymax - YY])
    clearance = border.copy()
    for o in obstacles:
        d = np.sqrt((XX - o.x)**2 + (YY - o.y)**2) - o.r
        clearance = np.minimum(clearance, d)

    # A cell is reachable if a robot CENTER there keeps STATIC_CLEAR from surfaces.
    reachable = clearance >= STATIC_CLEAR

    # Keep only the connected component that links the bottom end to the top end.
    lbl, n = ndimage.label(reachable)
    def end_label(y_target):
        j = int(round((y_target - ymin)/cell)); j = min(max(j,0),ny-1)
        row = lbl[j]
        vals = row[row>0]
        return vals[len(vals)//2] if vals.size else 0
    lab_bot = end_label(1.0); lab_top = end_label(MAP_H-1.0)
    main = np.zeros_like(reachable)
    if lab_bot and lab_bot==lab_top:
        main = (lbl==lab_bot)
    elif lab_bot:
        main = (lbl==lab_bot)

    # TRUE corridor width: for a longitudinal corridor, the passage width at a
    # given height y is the x-extent of the main reachable region in that row,
    # NOT the point-to-surface clearance (which would mislabel every wall-adjacent
    # cell as "narrow"). For each row, take the widest contiguous reachable x-run;
    # cells in a row narrower than the threshold are flagged narrow.
    width = np.full((ny, nx), 1e9)          # per-cell corridor width = its row's x-span
    row_span = np.zeros(ny)
    for j in range(ny):
        rowmask = main[j]
        if not rowmask.any():
            row_span[j] = 0.0
            width[j, :] = 0.0
            continue
        # widest contiguous run of True
        best = cur = 0
        for v in rowmask:
            cur = cur + 1 if v else 0
            if cur > best: best = cur
        row_span[j] = best * cell
        width[j, :] = best * cell
    # Narrow cells: inside main passage AND their row's corridor width < threshold.
    narrow = main & (width < GAP_WIDTH_THRESHOLD)

    grid_meta = (xmin, ymin, nx, ny, cell)
    bottlenecks = []; reroutable = []; narrow_centers = []

    if main.any() and lab_bot and lab_bot==lab_top:
        # Cluster narrow cells into gap regions.
        nlbl, nn = ndimage.label(narrow)
        bot_j = min(max(int(round((1.0-ymin)/cell)),0),ny-1)
        top_j = min(max(int(round((MAP_H-1.0-ymin)/cell)),0),ny-1)
        def connected_excluding(mask_remove):
            m = main & ~mask_remove
            l2, _ = ndimage.label(m)
            rb = l2[bot_j][m[bot_j]] if m[bot_j].any() else np.array([])
            rt = l2[top_j][m[top_j]] if m[top_j].any() else np.array([])
            return rb.size>0 and rt.size>0 and len(set(rb.tolist()) & set(rt.tolist()))>0
        for k in range(1, nn+1):
            comp = (nlbl==k)
            cy = YY[comp].mean(); cx = XX[comp].mean()
            narrow_centers.append((cx, cy))
            # Dilate the narrow region slightly so removal fully severs it.
            grown = ndimage.binary_dilation(comp, iterations=3)
            if not connected_excluding(grown):
                wait = [(cx, max(ymin+1.0, cy-1.8)), (cx, min(ymax-1.0, cy+1.8))]
                wmin = width[comp].min()
                bottlenecks.append({"pos": (float(cx), float(cy)),
                                     "wait": [(float(w[0]),float(w[1])) for w in wait],
                                     "min_width": float(wmin)})
            else:
                reroutable.append((float(cx), float(cy)))

    return {
        "reachable": reachable,
        "main": main,
        "width": width,
        "narrow_centers": narrow_centers,
        "bottlenecks": bottlenecks,
        "reroutable_gaps": reroutable,
        "grid_meta": grid_meta,
        "connected": bool(lab_bot and lab_bot==lab_top),
    }


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
class Robot:
    def __init__(self, rid, x, y, theta):
        self.id = rid
        self.x = x; self.y = y; self.theta = theta
        self.goal = None
        self.v = 0.0; self.w = 0.0
        self.locked = False          # deadlocked -> becomes a static obstacle
        self.collided = False
        self.tasks_done = 0
        # Soft-lock recovery: instead of locking forever, a deadlocked robot sleeps
        # for RECOVER_SLEEP steps (still a static obstacle so others don't hit it),
        # then revives and retries its original goal. If the environment changed
        # (a neighbor moved away) it proceeds; otherwise it locks and sleeps again.
        self.lock_timer = 0          # steps remaining asleep (>0 while locked)
        self.lock_attempts = 0       # how many times it has soft-locked (for telemetry)
        # Progress watchdog (#14): the single non-bypassable deadlock signal.
        # Tracks the closest the robot has gotten to its CURRENT goal, and how
        # long since it last meaningfully improved on that. No state (yielding,
        # turning, token-holding) is exempt -- if a robot hasn't gotten closer
        # to a goal for STALL_TIMEOUT steps, it is stuck, period.
        self.best_goal_dist = None   # closest distance achieved to current goal
        self.stall_steps = 0         # steps since best_goal_dist last improved
        self.pos_long = deque(maxlen=STALL_TIMEOUT)  # long window for absolute net-displacement floor (#14)
        # State-gated deadlock detection (做法B): deadlock check only runs when
        # the robot is SUPPOSED to be moving toward a goal. `waiting` is set True
        # by the initial-zone release scheduler (queued, not yet released) or by
        # the coordination layer (holding at a wait zone for a token). While
        # waiting==True, no-progress is legitimate and never counts as deadlock.
        self.waiting = True          # starts queued in the initial zone
        self.released = False        # released from initial zone to start tasks
        self.start_pos = np.array([x, y])
        self.path = None             # polyline waypoints from global planner
        self.wp_idx = 0
        self._turning = False        # hysteresis state for stop-turn-go
        self.reached = []            # list of completed-task goal points (x,y)
        self.coord_target = None     # coordination-layer retreat target (or None)
        self.coord_wait = 0.0        # accrued wait time for token priority
        self.coord_committed = {}    # bottleneck_idx -> committed (token locked)
        self.coord_start_side = {}   # bottleneck_idx -> side index at engage time
        self.coord_pos5 = deque(maxlen=50)   # last 5 s of positions (50 steps) for jam detection
        self.coord_dyn_hold = False  # holds the dynamic (jam) token
        self.coord_retreating = False  # currently retreating along path back to its start
        self._hist = deque(maxlen=DEADLOCK_PATIENCE)

    def pos(self):
        return np.array([self.x, self.y])


# ---------------------------------------------------------------------------
# ORCA velocity computation (2D holonomic preferred velocity -> safe velocity),
# then mapped to differential-drive (v, w). Standard ORCA half-plane + a small
# linear program (we use a simple, robust projection / sampling fallback).
# ---------------------------------------------------------------------------
def _orca_preferred(robot):
    """Preferred holonomic velocity. If the coordination layer set a temporary
    target (a wait point to retreat to), aim there; otherwise aim at the current
    waypoint toward the goal. Capped at V_MAX."""
    target = getattr(robot, "coord_target", None)
    if target is None:
        target = _current_waypoint(robot)
    if target is None:
        return np.array([0.0, 0.0])
    d = np.asarray(target, dtype=float) - robot.pos()
    dist = np.linalg.norm(d)
    if dist < 1e-6:
        return np.array([0.0, 0.0])
    speed = min(V_MAX, dist / DT)
    return d / dist * speed


def _orca_velocity(robot, neighbors, static_discs, tau=NEIGHBOR_HORIZON):
    """Collision-avoiding holonomic velocity. Vectorized candidate scoring:
    all candidates x all obstacles evaluated with NumPy for speed. ORCA-style
    velocity-obstacle (time-to-collision within tau) plus a mild close-range
    repulsion. Static obstacles use a smaller clearance than dynamic robots.
    """
    pref = _orca_preferred(robot)
    p = robot.pos()

    # Stack obstacles: positions (M,2), velocities (M,2), combined radii (M,)
    OP = []; OV = []; CR = []
    for nb in neighbors:
        OP.append([nb.x, nb.y])
        OV.append([nb.v * math.cos(nb.theta), nb.v * math.sin(nb.theta)])
        CR.append(ORCA_RADIUS + ROBOT_RADIUS * SAFETY_FACTOR)  # dynamic
    for d in static_discs:
        if (d.x - robot.x)**2 + (d.y - robot.y)**2 > 9.0:  # ignore beyond 3 m
            continue
        OP.append([d.x, d.y]); OV.append([0.0, 0.0])
        CR.append(STATIC_CLEAR + d.r)                          # static
    if not OP:
        return pref
    OP = np.asarray(OP); OV = np.asarray(OV); CR = np.asarray(CR)

    # Candidate velocities (N,2): preferred + ring samples (speed x heading)
    speeds = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * V_MAX
    headings = np.linspace(0, 2*math.pi, 24, endpoint=False)
    ring = np.array([[s*math.cos(h), s*math.sin(h)] for s in speeds for h in headings])
    cands = np.vstack([pref[None, :], ring])           # (N,2)

    rel_p = OP[None, :, :] - p[None, None, :]          # (1,M,2)
    rel_p = np.broadcast_to(rel_p, (cands.shape[0], OP.shape[0], 2))
    rel_v = cands[:, None, :] - OV[None, :, :]         # (N,M,2)
    dist = np.linalg.norm(OP - p, axis=1)              # (M,)

    pen = np.zeros(cands.shape[0])

    # Close-range repulsion (mild; only very near surfaces), per obstacle, applies
    # equally to all candidates -> constant offset, doesn't distort speed choice.
    near = np.maximum(0.0, (CR + 0.2) - dist)
    pen += np.sum(near**2 * 4.0)

    # Hard overlap push
    overlap = dist < (CR + 1e-3)

    # Time to closest approach within [0, tau]
    rv_dot = np.sum(rel_v * rel_p, axis=2)             # (N,M)
    vv = np.sum(rel_v * rel_v, axis=2)                 # (N,M)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(vv > 1e-9, rv_dot / vv, 0.0)
    t = np.clip(t, 0.0, tau)
    closest = rel_p - rel_v * t[:, :, None]            # (N,M,2)
    cd = np.linalg.norm(closest, axis=2)               # (N,M)
    closing = rv_dot > 0
    violate = closing & (cd < CR[None, :])
    vo_pen = np.where(violate, (CR[None, :] - cd) * (1.0 + (tau - t)), 0.0)
    vo_pen[:, overlap] += 1000.0
    pen += np.sum(vo_pen, axis=1)

    dev = np.linalg.norm(cands - pref[None, :], axis=1)
    idle = np.where((np.linalg.norm(cands, axis=1) < 0.05) &
                    (np.linalg.norm(pref) > 0.05), 0.5, 0.0)
    score = pen * 50.0 + dev + idle
    best = cands[int(np.argmin(score))]
    return best



def _to_diff_drive(robot, vel):
    """Map a desired holonomic velocity to differential-drive (v, w) honoring
    R_min (|w| <= v / R_min => |w| <= W_MAX scaled) and v_max."""
    speed = np.linalg.norm(vel)
    if speed < 1e-6:
        return 0.0, 0.0
    desired_theta = math.atan2(vel[1], vel[0])
    err = math.atan2(math.sin(desired_theta - robot.theta),
                     math.cos(desired_theta - robot.theta))
    # angular velocity proportional to heading error, capped
    w = max(-W_MAX, min(W_MAX, 2.0 * err))
    # reduce linear speed when we must turn sharply (respect turning radius)
    align = max(0.0, math.cos(err))
    v = min(V_MAX, speed) * align
    # enforce |w| <= v / R_min when moving; if nearly stopped allow turn in place
    if v > 1e-3 and abs(w) > v / R_MIN:
        w = math.copysign(v / R_MIN, w)
    return v, w


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def step_world(robots, walls, obstacles, annotation, rng, bounds, coordinator=None):
    """Advance the world one tick. Locked robots act as static obstacles."""
    # --- Soft-lock recovery (#12) -------------------------------------------
    # A soft-locked robot sleeps (stays a static obstacle) for a while, then
    # revives and retries its goal. By the time it wakes, neighbors may have
    # moved, so the standoff that locked it can resolve on its own. Only after
    # too many failed revivals do we leave it hard-locked.
    for r in robots:
        if r.locked and not r.collided:
            if r.lock_attempts >= HARD_LOCK_ATTEMPTS:
                continue            # genuine hard lock -> stays a static obstacle
            r.lock_timer -= 1
            if r.lock_timer <= 0:
                r.locked = False    # revive: rejoin active set, retry the goal
                r.waiting = False
                r._turning = False
                r.coord_wait = 0.0
                r._hist.clear()
                r.best_goal_dist = None   # fresh progress baseline on revival (#14)
                r.stall_steps = 0
                r.pos_long.clear()
                plan_path(r, obstacles, bounds)   # fresh path from current pos
    # --- Coordination layer (optional, on top of ORCA) ----------------------
    # Decides right-of-way at annotated bottlenecks. It only sets each robot's
    # effective target (coord_target) and waiting flag; ORCA still avoids.
    if coordinator is not None:
        decisions = coordinator.step([r for r in robots if r.released], DT)
        for r in robots:
            d = decisions.get(r.id, ("free", None))
            if d[0] == "wait":
                r.coord_target = np.asarray(d[1], dtype=float)
                r.waiting = True            # legitimate wait -> deadlock-exempt
            elif d[0] == "retreat":
                # Reverse along OWN path, stopping at the first point that is
                # clear of the holder's shared path (A: shared-path aware yield).
                r.waiting = True
                hp = getattr(r, "_holder_path", None)
                CLEAR_DIST = 2 * ORCA_RADIUS + 0.3   # must clear the holder's lane
                if getattr(r, "path", None):
                    if not hasattr(r, "_retreat_idx") or r._retreat_idx is None:
                        r._retreat_idx = max(0, r.wp_idx - 1)
                    # The robot may have switched goals (and thus replanned a
                    # fresh, shorter path) mid-retreat; clamp the stale index so
                    # it never points past the current path. (#5)
                    r._retreat_idx = min(r._retreat_idx, len(r.path) - 1)
                    def _clear(pt):
                        if not hp:
                            return False
                        return min(math.hypot(pt[0]-q[0], pt[1]-q[1]) for q in hp) > CLEAR_DIST
                    # walk index down until the point is clear of holder's path
                    tgt = np.asarray(r.path[r._retreat_idx], dtype=float)
                    while r._retreat_idx > 0 and not _clear(r.path[r._retreat_idx]):
                        r._retreat_idx -= 1
                    tgt = np.asarray(r.path[r._retreat_idx], dtype=float)
                    r.coord_target = tgt
                else:
                    r.coord_target = np.asarray(r.start_pos, dtype=float)
            else:                            # 'go' or 'free'
                r.coord_target = None
                r._retreat_idx = None
                if r.released and not r.locked:
                    r.waiting = False
    # --- Initial-zone release scheduler -------------------------------------
    # Robots start queued (waiting=True). Release the next queued robot only
    # once the previously released robots have cleared the launch area, so the
    # narrow corridor is never entered by two robots simultaneously at start.
    RELEASE_CLEAR_DIST = 2.4  # a robot must move this far from its start to free the next
    queued = [r for r in robots if not r.released and not r.collided]
    queued.sort(key=lambda r: r.id)
    if queued:
        any_in_launch = any(r.released and
                            np.linalg.norm(r.pos() - r.start_pos) <= RELEASE_CLEAR_DIST
                            for r in robots)
        if not any_in_launch:
            nxt = queued[0]
            nxt.released = True
            nxt.waiting = False
            plan_path(nxt, obstacles, bounds)

    active = [r for r in robots if not r.locked and not r.collided and r.released]
    # Locked/collided/queued robots act as static obstacles, but track them
    # SEPARATELY from real walls/obstacles: a frozen robot is a *robot*, not a
    # wall, so the dynamic token holder should be allowed to squeeze past it with
    # only near-physical clearance (just like it does for active yielders). If we
    # lumped them into all_static they would use the larger STATIC_CLEAR margin
    # (1.15 m) and actually block the holder MORE than a moving yielder does. (#15)
    frozen_discs = []  # locked/collided/queued robots (frozen, pure static obstacles)
    for r in robots:
        if r.locked or r.collided or not r.released:
            frozen_discs.append(Disc(r.x, r.y, ROBOT_RADIUS))
    all_static = walls + obstacles + frozen_discs

    # Compute velocities (read-only neighbor snapshot = bulletin board)
    cmds = {}
    for r in active:
        # Token holder right-of-way: a robot holding the dynamic (jam) token
        # treats other robots NOT with the full conservative dynamic margin
        # (which makes it stall and the yield do nothing), but as small
        # near-physical obstacles -- enough to avoid an actual collision, yet
        # tight enough to drive through the gap the yielders open up. Walls and
        # static obstacles are still avoided normally.
        if getattr(r, "coord_dyn_hold", False):
            neighbors = []
            # Walls + real obstacles: avoid normally (full static clearance).
            holder_static = list(walls) + list(obstacles)
            # Active robots AND frozen (locked) robots alike: treat as small
            # near-physical obstacles so the holder can thread the gap the
            # yielders open up -- and equally drive past a frozen car, which is
            # exactly what the locked-as-static-obstacle design is meant to allow.
            for o in active:
                if o.id != r.id:
                    holder_static.append(Disc(o.x, o.y, ROBOT_RADIUS + 0.1))
            # A frozen robot cannot move out of the way, so the holder must keep
            # enough margin to never physically overlap it. The collision check
            # fires below 2*ROBOT_RADIUS - 0.05 = 0.95 m center-to-center, so the
            # holder's center must stay above that. We set the effective obstacle
            # radius (= min center-to-center gap ORCA enforces) just above it.
            # This is still far tighter than the normal static margin (1.15 m),
            # so the holder passes close but never grazes the frozen car. (#15)
            FROZEN_PASS_CLEAR = (2 * ROBOT_RADIUS - 0.05) + 0.1   # 1.05 m center gap
            for fd in frozen_discs:
                holder_static.append(Disc(fd.x, fd.y, FROZEN_PASS_CLEAR))
            vel = _orca_velocity(r, neighbors, holder_static)
        else:
            neighbors = [o for o in active if o.id != r.id]
            # ORCA returns an avoidance-adjusted desired direction (holonomic).
            vel = _orca_velocity(r, neighbors, all_static)
        # Architecture 1: convert via stop-turn-go follower (path-trackable).
        if STEER_MODE == "arch1":
            cmds[r.id] = _steer_arch1(r, vel)
        elif STEER_MODE == "arch2":
            cmds[r.id] = _steer_arch2(r, vel)
        else:
            cmds[r.id] = _to_diff_drive(r, vel)

    # Apply motion
    for r in active:
        v, w = cmds[r.id]
        r.v, r.w = v, w
        r.theta += w * DT
        r.x += v * math.cos(r.theta) * DT
        r.y += v * math.sin(r.theta) * DT
        r._hist.append((r.x, r.y))

    # Advance waypoint when reached. For dense (arch2) curves, skip all points
    # within a lookahead radius so the robot aims ahead along the curve.
    LOOKAHEAD = 0.6 if STEER_MODE == "arch2" else GOAL_TOL
    for r in active:
        if getattr(r, "path", None):
            while r.wp_idx < len(r.path):
                wp = np.array(r.path[r.wp_idx])
                if np.linalg.norm(wp - r.pos()) < LOOKAHEAD:
                    r.wp_idx += 1
                else:
                    break

    # Collision check (physical radius). Only released, moving robots can be in
    # collision; ignore the spawn frame and use a clean physical threshold.
    COLL = 2 * ROBOT_RADIUS
    for i, a in enumerate(robots):
        if a.collided or not a.released:
            continue
        for b in robots[i+1:]:
            if b.collided or not b.released:
                continue
            if math.hypot(a.x-b.x, a.y-b.y) < COLL - 0.05:
                a.collided = b.collided = True
        for d in (walls + obstacles):
            if math.hypot(a.x-d.x, a.y-d.y) < ROBOT_RADIUS + d.r - 0.05:
                a.collided = True

    # Goal reached -> log it, assign a new random goal (endless pick-and-place loop)
    for r in active:
        if r.goal is not None and np.linalg.norm(r.goal - r.pos()) < GOAL_TOL:
            r.reached.append((float(r.goal[0]), float(r.goal[1])))  # completed task point
            r.tasks_done += 1
            r.lock_attempts = 0     # made real progress -> reset recovery budget (#12)
            r.best_goal_dist = None # new goal -> reset progress watchdog (#14)
            r.stall_steps = 0
            r.pos_long.clear()
            r.goal = sample_goal(rng, obstacles)
            plan_path(r, obstacles, bounds)

    # Deadlock detection (#14): progress watchdog -- the single, non-bypassable
    # signal. For every released robot with a goal, we track the closest it has
    # ever gotten to that goal. Each step that fails to improve on that closest
    # distance by at least PROGRESS_DELTA increments a stall counter; any genuine
    # improvement resets it. If the counter reaches STALL_TIMEOUT the robot has
    # demonstrably failed to advance toward its goal and is declared deadlocked.
    #
    # No state is exempt. A yielding, turning, or token-holding robot that is
    # actually making progress will keep getting closer to SOME goal and so will
    # never trip the watchdog; one that only looks busy (circling, oscillating,
    # retreating forever, stalled by ORCA) makes no progress and WILL trip it.
    # This is what guarantees no silent stuck state slips through.
    for r in active:
        if r.goal is None:
            continue
        d = float(np.linalg.norm(r.goal - r.pos()))
        if r.best_goal_dist is None or d < r.best_goal_dist - PROGRESS_DELTA:
            r.best_goal_dist = d        # real progress toward the goal
            r.stall_steps = 0
        else:
            r.stall_steps += 1
        # Second leg: absolute net-displacement floor over a long window. Even if
        # the goal-distance test is fooled by slow drift/oscillation that
        # occasionally nicks the best distance, a robot that physically goes
        # nowhere over STALL_TIMEOUT steps (net move < 0.5 m) is stuck. This makes
        # the watchdog robust to "fake progress" wobble. (#14)
        r.pos_long.append((r.x, r.y))
        net_long = 0.0
        if len(r.pos_long) >= STALL_TIMEOUT:
            net_long = math.hypot(r.pos_long[-1][0]-r.pos_long[0][0],
                                  r.pos_long[-1][1]-r.pos_long[0][1])
        frozen = (len(r.pos_long) >= STALL_TIMEOUT and net_long < 0.5)
        # A robot legitimately waiting in a queue (not yet released) is handled by
        # `active` excluding it; released robots that merely wait at a coord point
        # are NOT exempt -- if that wait never lets them progress, it IS a deadlock.
        if r.stall_steps >= STALL_TIMEOUT or frozen:
            # Soft lock: sleep, then revive and retry (#12). Hard lock only after
            # repeated failed revivals.
            r.locked = True
            r.lock_timer = RECOVER_SLEEP
            r.lock_attempts += 1
            r.stall_steps = 0
            r.best_goal_dist = None     # fresh progress baseline after revival
            r.pos_long.clear()


def sample_goal(rng, obstacles, max_try=400):
    """Random goal inside the REACHABLE region only. A band along every wall and
    around every obstacle is unreachable by design (to prevent wall collisions),
    so goals must keep >= ORCA safety clearance from them; otherwise a robot
    would be sent to a point ORCA will never let it reach, and would lock up."""
    wall_clear = STATIC_CLEAR         # unreachable band thickness along walls
    for _ in range(max_try):
        gx = rng.uniform(wall_clear, MAP_W - wall_clear)
        gy = rng.uniform(wall_clear, MAP_H - wall_clear)
        ok = True
        for o in obstacles:
            if math.hypot(gx-o.x, gy-o.y) < o.r + STATIC_CLEAR + 0.1:
                ok = False; break
        if ok:
            return np.array([gx, gy])
    return np.array([MAP_W/2, MAP_H/2])


def run_sim(seed=None, max_steps=None, record=True):
    seed = int(os.environ.get("SEED", seed if seed is not None else np.random.randint(1, 1_000_000)))
    max_steps = int(os.environ.get("MAX_STEPS", max_steps if max_steps is not None else 10000))
    rng = np.random.default_rng(seed)

    walls, obstacles, bounds = make_container_map(rng)
    annotation = annotate_map(obstacles, bounds)
    coordinator = coord_mod.Coordinator(annotation) if COORD else None

    # Initial zone: random placement (Poisson-disc) inside a bottom rectangle,
    # enforcing pairwise spacing >= D_DEPLOY (>= d_min = 2*ORCA_RADIUS = 2.0 m).
    # Headings random. Horizontal-and-vertical staggering emerges naturally, so
    # no queued robot structurally blocks another's departure direction.
    INIT_X = (1.0, 4.0); INIT_Y = (1.0, 7.0)   # taller zone so 3 robots always fit
    D_DEPLOY = 2.2                               # >= d_min = 2.0 m
    starts = []
    tries = 0
    while len(starts) < 3 and tries < 8000:
        tries += 1
        px = rng.uniform(*INIT_X); py = rng.uniform(*INIT_Y)
        if all(math.hypot(px-q[0], py-q[1]) >= D_DEPLOY for q in starts):
            if all(math.hypot(px-o.x, py-o.y) > o.r + ROBOT_RADIUS + 0.2 for o in obstacles):
                starts.append((px, py))
    # Deterministic fallback: stagger along the centerline if sampling fell short.
    fb_y = 1.5
    while len(starts) < 3:
        cand = (MAP_W/2.0, fb_y)
        if all(math.hypot(cand[0]-q[0], cand[1]-q[1]) >= D_DEPLOY for q in starts) and \
           all(math.hypot(cand[0]-o.x, cand[1]-o.y) > o.r + ROBOT_RADIUS + 0.2 for o in obstacles):
            starts.append(cand)
        fb_y += D_DEPLOY
        if fb_y > MAP_H - 1.0:
            starts.append((MAP_W/2.0, min(fb_y, MAP_H-1.0)))  # last resort
    robots = []
    for i, (x0, y0) in enumerate(starts):
        th0 = rng.uniform(0, 2*math.pi)
        r = Robot(i, x0, y0, th0)
        r.goal = sample_goal(rng, obstacles)
        # Safely spaced at start -> release immediately; ORCA handles the rest.
        r.released = True
        r.waiting = False
        plan_path(r, obstacles, bounds)
        robots.append(r)

    trajectories = [[] for _ in robots]
    goal_track = [[] for _ in robots]   # current goal each step (for animation)
    step = 0
    stop_reason = "max_steps"
    while step < max_steps:
        step_world(robots, walls, obstacles, annotation, rng, bounds, coordinator)
        if record:
            for i, r in enumerate(robots):
                trajectories[i].append((r.x, r.y, r.theta, r.locked, r.collided))
                goal_track[i].append((float(r.goal[0]), float(r.goal[1])) if r.goal is not None else None)
        if any(r.collided for r in robots):
            stop_reason = "collision"
            break
        # With soft-lock recovery, a momentarily all-locked fleet may still revive.
        # Only stop when every robot is HARD-locked (recovery attempts exhausted)
        # or collided -- a genuine, unrecoverable deadlock. (#12)
        if all((r.locked and r.lock_attempts >= HARD_LOCK_ATTEMPTS) or r.collided
               for r in robots):
            stop_reason = "all_locked"
            break
        step += 1

    return {
        "seed": seed, "steps": step, "stop_reason": stop_reason,
        "robots": robots, "walls": walls, "obstacles": obstacles,
        "bounds": bounds, "annotation": annotation, "trajectories": trajectories,
        "tasks_done": [r.tasks_done for r in robots],
        "reached": [list(r.reached) for r in robots],   # completed task points
        "goal_track": goal_track,                         # per-step current goal
    }


# ===========================================================================
# GLOBAL PLANNER LAYER (Architecture 1: polyline path + stop-turn-go follower)
# ===========================================================================
# Per project narrative: a high-level planner produces a geometric PATH (a
# sequence of straight waypoints). The local layer (ORCA) follows the path while
# avoiding dynamic neighbors. The diff-drive follower uses "stop-turn-go" so the
# executed trajectory matches the straight path (path trackability), with a small
# (<15 deg) tolerance allowing arc-while-driving for minor avoidance (做法乙).
# STEER_MODE leaves a hook for Architecture 2 (Dubins/spline) later.

STEER_MODE = os.environ.get("STEER_MODE", "arch1")  # "arch1" | "arch2"(reserved)
TURN_TOL_DEG = 15.0                 # below this heading error -> drive (with micro-correct)
ALIGN_DONE_DEG = 5.0                # consider aligned when within this
NEAR_TARGET_DIST = 1.0              # within this range of the target, arc in instead of
                                    # stop-and-pivot, to avoid the limit-cycle oscillation (#11)

def _astar_path(start, goal, obstacles, bounds, cell=0.2):
    """A* on an inflated-free grid, then line-of-sight simplification to a
    minimal set of straight waypoints. Returns list of (x,y) incl. goal."""
    xmin, xmax, ymin, ymax = bounds
    nx = int(round((xmax - xmin) / cell)) + 1
    ny = int(round((ymax - ymin) / cell)) + 1

    def free_cell(i, j):
        px = xmin + i*cell; py = ymin + j*cell
        # Reachable region = ORCA safety clearance from walls...
        if not (STATIC_CLEAR <= px <= MAP_W-STATIC_CLEAR and
                STATIC_CLEAR <= py <= MAP_H-STATIC_CLEAR):
            return False
        # ...and from every obstacle.
        for o in obstacles:
            if math.hypot(px-o.x, py-o.y) < o.r + STATIC_CLEAR + 0.1:
                return False
        return True

    def to_cell(p):
        return (int(round((p[0]-xmin)/cell)), int(round((p[1]-ymin)/cell)))
    def to_world(c):
        return (xmin + c[0]*cell, ymin + c[1]*cell)

    s = to_cell(start); g = to_cell(goal)
    # clamp into grid
    s = (min(max(s[0],0),nx-1), min(max(s[1],0),ny-1))
    g = (min(max(g[0],0),nx-1), min(max(g[1],0),ny-1))

    import heapq
    def h(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
    openq = [(h(s,g), 0.0, s)]
    came = {}; gscore = {s: 0.0}
    found = False
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == g:
            found = True; break
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0: continue
                nb = (cur[0]+dx, cur[1]+dy)
                if not (0<=nb[0]<nx and 0<=nb[1]<ny): continue
                if not free_cell(*nb): continue
                step = math.hypot(dx,dy)
                ng = gc + step
                if ng < gscore.get(nb, 1e18):
                    gscore[nb] = ng; came[nb] = cur
                    heapq.heappush(openq, (ng + h(nb,g), ng, nb))
    if not found:
        return [tuple(goal)]  # fallback: head straight (planner failed)

    # reconstruct
    path = [g]
    while path[-1] in came:
        path.append(came[path[-1]])
    path.reverse()
    pts = [to_world(c) for c in path]

    # line-of-sight simplification
    def los(a, b):
        d = math.hypot(b[0]-a[0], b[1]-a[1])
        n = max(2, int(d/ (cell*0.8)))
        for k in range(n+1):
            t = k/n
            px = a[0] + (b[0]-a[0])*t; py = a[1] + (b[1]-a[1])*t
            for o in obstacles:
                if math.hypot(px-o.x, py-o.y) < o.r + STATIC_CLEAR + 0.1:
                    return False
            if not (STATIC_CLEAR <= px <= MAP_W-STATIC_CLEAR and
                    STATIC_CLEAR <= py <= MAP_H-STATIC_CLEAR):
                return False
        return True

    simplified = [pts[0]]
    i = 0
    while i < len(pts)-1:
        j = len(pts)-1
        while j > i+1 and not los(pts[i], pts[j]):
            j -= 1
        simplified.append(pts[j])
        i = j
    # ensure exact goal at end
    simplified[-1] = tuple(goal)
    return simplified


def plan_path(robot, obstacles, bounds):
    """Assign a fresh path toward the goal. In arch1 the path is the A* polyline
    (followed by stop-turn-go). In arch2 the polyline is rounded into a smooth,
    minimum-turning-radius curve that the robot tracks while edge-turning."""
    poly = _astar_path((robot.x, robot.y), tuple(robot.goal), obstacles, bounds)
    if STEER_MODE == "arch2" and len(poly) >= 2:
        robot.path = _round_corners(poly, R_MIN)
    else:
        robot.path = poly
    robot.wp_idx = 0
    # New path invalidates any in-progress retreat index (it referenced the
    # old polyline's waypoint numbering). (#5)
    robot._retreat_idx = None


def _current_waypoint(robot):
    if not getattr(robot, "path", None):
        return robot.goal
    if robot.wp_idx >= len(robot.path):
        return robot.goal
    return np.array(robot.path[robot.wp_idx])


def _steer_arch1(robot, desired_vec):
    """Stop-turn-go follower with hysteresis (做法乙: arc-while-driving for small
    corrections, in-place turn only for large reorientation). ORCA still chooses
    the DIRECTION (so genuine side-stepping / detours are preserved); this only
    governs HOW the diff-drive realizes that direction and at what speed.

    Speed is decoupled from |desired_vec|: when roughly aligned and the path
    ahead is not strongly compressed we drive near V_MAX; we only slow down when
    ORCA has shrunk the desired velocity a lot (i.e. avoidance truly requires it).
    """
    speed_des = np.linalg.norm(desired_vec)
    if speed_des < 1e-6:
        # ORCA wants a full stop (blocked) -> hold position (time-avoidance).
        return 0.0, 0.0
    desired_theta = math.atan2(desired_vec[1], desired_vec[0])
    err = math.atan2(math.sin(desired_theta - robot.theta),
                     math.cos(desired_theta - robot.theta))
    err_deg = abs(math.degrees(err))

    # Distance to the active target (goal or coord wait point). Near the target,
    # each forward step swings the bearing sharply, so the fixed turn threshold
    # makes the robot oscillate: align -> step -> overshoot -> turn back -> repeat
    # (a limit cycle that pins it in place and trips false deadlock). (#11)
    tgt = getattr(robot, "coord_target", None)
    if tgt is None:
        tgt = robot.goal
    near_target = (tgt is not None and
                   np.linalg.norm(np.asarray(tgt, float) - robot.pos()) < NEAR_TARGET_DIST)

    # Hysteresis: enter in-place turning only on a LARGE error; once turning,
    # keep turning until well aligned. This removes chatter around a single
    # threshold when ORCA's chosen direction wiggles slightly each tick.
    ENTER_TURN = 30.0
    EXIT_TURN  = 10.0
    if robot._turning:
        if err_deg < EXIT_TURN:
            robot._turning = False
    else:
        if err_deg > ENTER_TURN:
            robot._turning = True

    # Near the target, never stop-and-pivot for moderate errors -> arc in instead,
    # so position keeps advancing and the limit cycle cannot form. Only a near
    # full reversal (>120 deg) still justifies pivoting in place. (#11)
    if near_target and err_deg < 120.0:
        robot._turning = False

    if robot._turning:
        # TURNING in place (v=0, free of R_min, fast W_TURN)
        return 0.0, math.copysign(W_TURN, err)

    # DRIVING: speed decoupled from |desired_vec|. Full speed scaled only by how
    # well we face the target (cos) and whether ORCA strongly compressed speed.
    align = max(0.0, math.cos(err))
    compress = min(1.0, speed_des / V_MAX)      # ~1 when ORCA didn't slow us
    v = V_MAX * align * (0.4 + 0.6 * compress)  # keep moving unless badly blocked
    if near_target:
        v = max(v, 0.25 * V_MAX * max(0.0, math.cos(err)))  # keep creeping in (#11)
    w = max(-W_MAX, min(W_MAX, 2.0 * err))      # arc-while-driving micro-correction
    if v > 1e-3 and abs(w) > v / R_MIN:         # respect R_min while moving
        w = math.copysign(v / R_MIN, w)
    return v, w


# ---------------------------------------------------------------------------
# ARCHITECTURE 2: curvature-continuous (Dubins) path + pure-pursuit follower
# ---------------------------------------------------------------------------
# The global planner's polyline is converted into a smooth, R_min-feasible curve
# by rounding each interior corner with a minimum-radius circular arc (a chained
# Dubins-style smoothing). The robot then DRIVES ALONG the curve continuously
# (v>0, edge-turning) using pure pursuit: it steers toward a lookahead point on
# the curve. The executed trajectory follows the planned curve, which itself is
# already feasible for the turning radius, so path and trajectory stay consistent
# AND motion is naturally smooth (no stop-and-turn).

def _round_corners(pts, radius, step=0.12):
    """Turn a polyline into a dense list of (x,y) samples where each interior
    vertex is replaced by a circular fillet of the given radius (clamped so it
    never exceeds half of either adjacent segment). Endpoints are preserved."""
    if len(pts) < 3:
        return _densify(pts, step)
    out = [tuple(pts[0])]
    for i in range(1, len(pts)-1):
        a = np.array(pts[i-1], float); b = np.array(pts[i], float); c = np.array(pts[i+1], float)
        v1 = a - b; v2 = c - b
        l1 = np.linalg.norm(v1); l2 = np.linalg.norm(v2)
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        u1 = v1/l1; u2 = v2/l2
        ang = math.acos(max(-1.0, min(1.0, np.dot(u1, u2))))   # interior angle
        if ang > math.radians(178):       # nearly straight -> no fillet
            out.append(tuple(b)); continue
        # fillet tangent distance from corner, clamped to half each segment
        t = radius / math.tan(ang/2.0)
        t = min(t, 0.5*l1, 0.5*l2)
        r_eff = t * math.tan(ang/2.0)
        p1 = b + u1*t      # tangent point on incoming segment
        p2 = b + u2*t      # tangent point on outgoing segment
        # straight part up to p1
        out += _densify([out[-1], tuple(p1)], step)[1:]
        # arc from p1 to p2 around the fillet center
        bis = u1 + u2; nb = np.linalg.norm(bis)
        if nb < 1e-6:
            out.append(tuple(p2)); continue
        bis /= nb
        center = b + bis * (r_eff/ math.sin(ang/2.0))
        a1 = math.atan2(p1[1]-center[1], p1[0]-center[0])
        a2 = math.atan2(p2[1]-center[1], p2[0]-center[0])
        # choose short arc direction
        da = a2 - a1
        while da > math.pi: da -= 2*math.pi
        while da < -math.pi: da += 2*math.pi
        n = max(2, int(abs(da)*r_eff/step))
        for k in range(1, n+1):
            aa = a1 + da*k/n
            out.append((center[0]+r_eff*math.cos(aa), center[1]+r_eff*math.sin(aa)))
    out += _densify([out[-1], tuple(pts[-1])], step)[1:]
    return out

def _densify(pts, step):
    out=[tuple(pts[0])]
    for i in range(1,len(pts)):
        a=np.array(pts[i-1],float); b=np.array(pts[i],float)
        d=np.linalg.norm(b-a); n=max(1,int(d/step))
        for k in range(1,n+1):
            out.append(tuple(a+(b-a)*k/n))
    return out

def _steer_arch2(robot, desired_vec):
    """Pure-pursuit along the smooth curve, blended with ORCA avoidance. Drives
    continuously while edge-turning, but eases speed toward ~0 when the heading
    error is large (so it effectively pivots instead of running off-course), then
    accelerates smoothly as it aligns. Curvature is bounded by R_min."""
    speed_des = np.linalg.norm(desired_vec)
    if speed_des < 1e-6:
        return 0.0, 0.0     # ORCA says stop (blocked)
    desired_theta = math.atan2(desired_vec[1], desired_vec[0])
    err = math.atan2(math.sin(desired_theta - robot.theta),
                     math.cos(desired_theta - robot.theta))
    err_deg = abs(math.degrees(err))

    if err_deg > 90:
        # Badly misaligned -> pivot in place (free of R_min), don't charge off.
        return 0.0, math.copysign(W_TURN, err)
    # Smooth edge-turning: full speed when aligned, fading to 0 near 90 deg.
    sf = math.cos(err)
    compress = min(1.0, speed_des / V_MAX)
    v = V_MAX * sf * (0.5 + 0.5*compress)
    if v > 1e-3:
        wmax = v / R_MIN                 # curvature-limited while driving
        w = max(-wmax, min(wmax, 2.5*err))
    else:
        w = math.copysign(W_TURN, err)
    return v, w

if __name__ == "__main__":
    res = run_sim()
    print(f"SEED={res['seed']}  steps={res['steps']}  stop={res['stop_reason']}")
    print(f"tasks_done per robot = {res['tasks_done']}  total={sum(res['tasks_done'])}")
    ann = res["annotation"]
    print(f"bottlenecks(unique-path)={len(ann['bottlenecks'])}  "
          f"reroutable_narrow_gaps={len(ann['reroutable_gaps'])}")
