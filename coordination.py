"""
Generalized distributed coordination layer (per-bottleneck priority mutex), on
TOP of ORCA. This is the container-scenario successor of the gate coordination
layer: same proven mechanism (global wait-time priority + commitment locking +
retreat-and-wait + two-layer separation), but GENERALIZED so it no longer needs
a hand-coded side_of() split. Instead it consumes the offline map annotation:

  annotation["bottlenecks"] = [ {"pos": (x,y), "wait": [(x,y), (x,y)], ...}, ... ]

Each bottleneck is a narrow AND unique-path passage; "the two sides" are simply
the two annotated wait-zone points, so the logic works for a passage of any
orientation (no x/y assumptions).

Mechanism (per bottleneck, independent token):
  * One token per bottleneck. A robot competes for a bottleneck's token only if
    its CURRENT PLANNED PATH actually crosses that bottleneck (participation
    rule 甲 -- checked from the A* waypoints, not from geometry guesses).
  * Token holder = the committed (already crossing) robot if any (commitment
    locking -> no token flicker); otherwise the contender with the largest
    accumulated wait time (ties broken by smaller id). Every robot computes the
    same holder from the shared board -> fully distributed, deterministic.
  * The holder drives through (ORCA still does avoidance). Non-holders retreat
    to the wait-zone point on THEIR side of the bottleneck (whichever annotated
    wait point is nearer) and hold there as a static obstacle, accruing wait.
  * When the holder has crossed to the far side, it leaves the competition and
    the next-highest contender takes the token.

This layer only sets each robot's effective target / waiting flag and right of
way; it never overrides ORCA's collision avoidance.
"""

import numpy as np

# ---- dynamic jam (congestion deadlock) detection + broadcast token ----------
JAM_WINDOW = 50          # steps (5 s at dt=0.1) over which to measure displacement
JAM_DISP = 0.4           # m: if a robot moves less than this over the window -> stalled
JAM_NEAR = 2.5           # m: two stalled robots within this distance -> a jam (facing off)
JAM_RETREAT = 1.6        # m: how far a yielding robot backs off along its incoming heading
JAM_CLEAR = 1.2          # m: holder is "clear" once this far from the jam centroid

# A robot is "near" a bottleneck (in its corridor zone) within this radius.
BOTTLENECK_ZONE = 1.6
# Distance past the bottleneck (toward the far wait point) at which we consider
# the holder to have fully crossed and can release the token.
CROSS_MARGIN = 0.6
# Tolerance for matching a path waypoint to the bottleneck (participation test).
PATH_CROSS_TOL = 1.0


def _nearer_wait_index(pos, wait_pts):
    """Index (0/1) of the wait point on the same side as pos."""
    d0 = np.hypot(pos[0]-wait_pts[0][0], pos[1]-wait_pts[0][1])
    d1 = np.hypot(pos[0]-wait_pts[1][0], pos[1]-wait_pts[1][1])
    return 0 if d0 <= d1 else 1


def path_crosses_bottleneck(path, bpos, tol=PATH_CROSS_TOL):
    """Participation rule 甲: does this planned polyline/curve pass through the
    bottleneck? True if any path point lies within tol of the bottleneck center."""
    if not path:
        return False
    for (px, py) in path:
        if np.hypot(px-bpos[0], py-bpos[1]) <= tol:
            return True
    return False


def robot_side(pos, wait_pts):
    """Which side of the bottleneck a position is on, named by nearer wait point."""
    return _nearer_wait_index(pos, wait_pts)


def has_crossed(pos, start_side, wait_pts):
    """True when the robot is now on the OTHER side from where it started
    (nearer to the opposite wait point). NOTE: this checks side only; the
    caller additionally requires the robot to be clear of BOTTLENECK_ZONE
    before releasing the token."""
    return robot_side(pos, wait_pts) != start_side


def token_holder(contenders):
    """contenders: list of dicts {id, wait, committed}. Returns holder id.
    Commitment locking first (a committed robot keeps the token); else max wait,
    ties -> smaller id. Mirrors the proven gate logic."""
    committed = [c for c in contenders if c.get("committed")]
    if committed:
        return min(c["id"] for c in committed)
    best, best_key = None, None
    for c in contenders:
        key = (c["wait"], -c["id"])
        if best_key is None or key > best_key:
            best_key, best = key, c["id"]
    return best


def retreat_point(pos, start_pos, wait_pts):
    """Where a non-holder waits: the wait point on its own side, unless its start
    is closer (less wasted travel) -- mirrors nearest_wait_point in the original."""
    side = _nearer_wait_index(pos, wait_pts)
    slot = np.asarray(wait_pts[side], dtype=float)
    start_pos = np.asarray(start_pos, dtype=float)
    if np.linalg.norm(np.asarray(pos)-slot) <= np.linalg.norm(np.asarray(pos)-start_pos):
        return slot
    return start_pos


class Coordinator:
    """Holds per-bottleneck state and, each tick, assigns right-of-way and
    waiting targets. Robots carry: .coord_wait (accrued), .coord_committed,
    .coord_start_side (per active bottleneck id)."""

    def __init__(self, annotation):
        self.bottlenecks = annotation.get("bottlenecks", [])

    def step(self, robots, dt):
        """Returns dict: robot_id -> ('go', None) | ('wait', point) | ('free', None).
        'go'   = holds a token / no contention, proceed to its own goal.
        'wait' = retreat to the given point and hold (waiting).
        'free' = not involved with any bottleneck, proceed normally."""
        decisions = {r.id: ("free", None) for r in robots}
        if not self.bottlenecks:
            return decisions

        for bi, bn in enumerate(self.bottlenecks):
            bpos = bn["pos"]; wait_pts = bn["wait"]

            # contenders: released, alive robots whose planned path crosses bi
            contenders = []
            for r in robots:
                if r.locked or r.collided or not r.released:
                    continue
                crossing = path_crosses_bottleneck(getattr(r, "path", None), bpos)
                # record start side the first time it engages this bottleneck
                if crossing and bi not in r.coord_start_side:
                    r.coord_start_side[bi] = robot_side(r.pos(), wait_pts)
                committed = r.coord_committed.get(bi, False)
                if crossing or committed:
                    # has it already crossed? then it's done with this bottleneck
                    ss = r.coord_start_side.get(bi, robot_side(r.pos(), wait_pts))
                    if committed and has_crossed(r.pos(), ss, wait_pts) and \
                       np.hypot(r.x-bpos[0], r.y-bpos[1]) > BOTTLENECK_ZONE:
                        r.coord_committed[bi] = False     # release token
                        continue
                    contenders.append({"id": r.id, "wait": r.coord_wait,
                                        "committed": committed, "robot": r})
            if not contenders:
                continue

            holder = token_holder(contenders)
            for c in contenders:
                r = c["robot"]
                if c["id"] == holder:
                    # commit the holder once it nears the bottleneck zone
                    if np.hypot(r.x-bpos[0], r.y-bpos[1]) < BOTTLENECK_ZONE:
                        r.coord_committed[bi] = True
                    decisions[r.id] = ("go", None)
                    r.coord_wait = 0.0      # holder resets its wait
                else:
                    # non-holder retreats and accrues wait time
                    decisions[r.id] = ("wait", retreat_point(r.pos(), r.start_pos, wait_pts))
                    r.coord_wait += dt

        # --- Dynamic jam handling (congestion deadlock anywhere) -------------
        # Update each robot's 5 s position window and detect stalled robots.
        for r in robots:
            if r.locked or r.collided or not r.released:
                continue
            r.coord_pos5.append((r.x, r.y))

        def stalled(r):
            if len(r.coord_pos5) < JAM_WINDOW:
                return False
            p0 = r.coord_pos5[0]; p1 = r.coord_pos5[-1]
            return np.hypot(p1[0]-p0[0], p1[1]-p0[1]) < JAM_DISP

        alive = [r for r in robots if not r.locked and not r.collided and r.released]
        stalled_robots = [r for r in alive if stalled(r)]

        # Group stalled robots that are near each other -> a jam.
        jam = []
        for i, a in enumerate(stalled_robots):
            for b in stalled_robots[i+1:]:
                if np.hypot(a.x-b.x, a.y-b.y) < JAM_NEAR:
                    if a not in jam: jam.append(a)
                    if b not in jam: jam.append(b)

        if len(jam) >= 2:
            # Broadcast token with ROTATION. Pick a holder by priority; it ignores
            # other-robot repulsion (handled in sim) and drives through. Once it has
            # pulled clear of the jam centroid, it RELEASES the token so the next
            # contender takes a turn -- this is what breaks a 3-way standoff that a
            # single permanent holder cannot (the others keep blocking each other).
            cx = float(np.mean([r.x for r in jam]))
            cy = float(np.mean([r.y for r in jam]))

            current = [r for r in jam if r.coord_dyn_hold]
            holder = None
            if current:
                h = min(current, key=lambda r: r.id)
                # Has the current holder pulled clear of the jam? If so, release it
                # (let it keep going as a normal robot) and rotate to the next one.
                if np.hypot(h.x - cx, h.y - cy) > JAM_CLEAR:
                    h.coord_dyn_hold = False
                    h.coord_wait = 0.0           # it got its turn; reset priority
                    current = []
                else:
                    holder = h                    # still crossing -> keep the token
            if holder is None:
                # Next turn goes to the longest-waiting contender that has NOT just
                # been released (its wait was zeroed), so turns actually rotate.
                contenders = [r for r in jam if not r.coord_dyn_hold]
                holder = max(contenders, key=lambda r: (r.coord_wait, -r.id))

            for r in jam:
                if r is holder:
                    r.coord_dyn_hold = True
                    r.coord_retreating = False
                    r.coord_wait = 0.0
                    decisions[r.id] = ("go", None)   # proceed to own goal
                else:
                    # YIELD: retreat back along OWN path, but only as far as needed
                    # to get clear of the HOLDER'S shared path (A: shared-path aware).
                    r.coord_dyn_hold = False
                    r.coord_retreating = True
                    r._holder_path = list(getattr(holder, "path", []) or [])
                    decisions[r.id] = ("retreat", None)
                    r.coord_wait += dt
        else:
            # No jam: release dynamic token; resume normal goal-seeking.
            for r in alive:
                r.coord_dyn_hold = False
                r.coord_retreating = False
                r._holder_path = None
        return decisions
