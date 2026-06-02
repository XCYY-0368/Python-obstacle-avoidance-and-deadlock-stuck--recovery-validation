"""Visualization for the container scenario: annotated map + trajectories + animation."""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation
import container_sim as cs

# Output directory for figures/animations. Configurable so this runs anywhere
# (the original hardcoded /home/claude/orca_demo/ only existed in one sandbox).
# Override with the VIZ_OUT env var; defaults to ./outputs next to this file.
OUT_DIR = os.environ.get("VIZ_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))


def _out(fname):
    """Resolve a filename to the output directory, creating it if needed."""
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, fname)


COLS = ["#1f77b4", "#2ca02c", "#d62728"]

def draw_static(ax, res, show_annotation=True):
    b = res["bounds"]
    ax.add_patch(Rectangle((0,0), cs.MAP_W, cs.MAP_H, fill=False, ec="black", lw=1.5))
    for o in res["obstacles"]:
        ax.add_patch(Circle((o.x,o.y), o.r, color="#8c6d4f", alpha=0.85, zorder=2))
    if show_annotation:
        for bn in res["annotation"]["bottlenecks"]:
            px,py = bn["pos"]
            ax.add_patch(Circle((px,py), 0.35, color="orange", alpha=0.5, zorder=3))
            ax.annotate("bottleneck\n(coord)", (px,py), color="darkorange",
                        fontsize=7, ha="center", va="center", zorder=5)
            for w in bn["wait"]:
                ax.plot(w[0], w[1], "x", color="orange", ms=8, mew=2, zorder=4)
    ax.set_xlim(-0.4, cs.MAP_W+0.4); ax.set_ylim(-0.4, cs.MAP_H+0.4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

def traj_panel(ax, res, title):
    draw_static(ax, res)
    for i, tr in enumerate(res["trajectories"]):
        if not tr: continue
        xs=[p[0] for p in tr]; ys=[p[1] for p in tr]
        ax.plot(xs, ys, color=COLS[i%3], lw=1.0, alpha=0.7, zorder=3)
        ax.plot(xs[0], ys[0], "o", color=COLS[i%3], ms=8, zorder=5, mec="white", mew=1.2)
        ax.plot(xs[-1], ys[-1], "s", color=COLS[i%3], ms=8, zorder=5,
                mec="red" if tr[-1][4] else "black", mew=1.5)
        # mark each completed task point with a numbered star
        for k, (gx, gy) in enumerate(res.get("reached", [[]]*3)[i], start=1):
            ax.plot(gx, gy, "*", color=COLS[i%3], ms=11, zorder=6, mec="white", mew=0.6)
            ax.annotate(str(k), (gx, gy), fontsize=6, color="white", ha="center",
                        va="center", zorder=7, fontweight="bold")
    ax.set_title(title, fontsize=9)

def grid_figure(seeds, max_steps=5000, fname="container_overview.png"):
    n=len(seeds)
    fig, axes = plt.subplots(1, n, figsize=(2.4*n, 8))
    if n==1: axes=[axes]
    for ax, s in zip(axes, seeds):
        os.environ["SEED"]=str(s); os.environ["MAX_STEPS"]=str(max_steps)
        res = cs.run_sim()
        t=sum(res["tasks_done"])
        reason={"all_locked":"DEADLOCK","collision":"COLLISION","max_steps":"survived"}[res["stop_reason"]]
        traj_panel(ax, res, "seed %s\n%d steps, %s\ntasks=%d"%(s,res["steps"],reason,t))
    fig.suptitle("Container scenario — pure ORCA + planner (Architecture 1)\n"
                 "circles=start, squares=end (red edge=collision), orange=annotated coordination bottleneck",
                 fontsize=10)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(_out(fname), dpi=90, bbox_inches="tight")
    print("saved", fname)

def make_animation(seed, max_steps=2000, fname=None):
    os.environ["SEED"]=str(seed); os.environ["MAX_STEPS"]=str(max_steps)
    res = cs.run_sim()
    traj = res["trajectories"]; gtrack = res.get("goal_track")
    nframes = res["steps"]
    fig, ax = plt.subplots(figsize=(3.4, 8.7))
    draw_static(ax, res)
    dots=[]; trails=[]; goals=[]
    for i in range(len(traj)):
        d,=ax.plot([],[],"o",color=COLS[i%3],ms=11,mec="white",mew=1.2,zorder=6)
        tl,=ax.plot([],[],"-",color=COLS[i%3],lw=1,alpha=0.5,zorder=3)
        g,=ax.plot([],[],"*",color=COLS[i%3],ms=14,mec="black",mew=0.6,zorder=5,alpha=0.9)
        dots.append(d); trails.append(tl); goals.append(g)
    txt=ax.text(0.02,0.99,"",transform=ax.transAxes,fontsize=8,va="top",
                bbox=dict(boxstyle="round",fc="white",ec="gray",alpha=0.8))
    # running task count = number of reached goals up to frame f (approx by goal changes)
    def upd(f):
        counts=[0,0,0]
        for i,tr in enumerate(traj):
            if f<len(tr):
                x,y,th,lk,co=tr[f]
                dots[i].set_data([x],[y])
                dots[i].set_color("red" if co else ("gray" if lk else COLS[i%3]))
                trails[i].set_data([p[0] for p in tr[:f+1]],[p[1] for p in tr[:f+1]])
                if gtrack and f<len(gtrack[i]) and gtrack[i][f]:
                    goals[i].set_data([gtrack[i][f][0]],[gtrack[i][f][1]])
                # count goal switches so far = completed tasks
                if gtrack:
                    seq=gtrack[i][:f+1]; c=0
                    for a,b in zip(seq,seq[1:]):
                        if a and b and (abs(a[0]-b[0])>1e-6 or abs(a[1]-b[1])>1e-6): c+=1
                    counts[i]=c
        txt.set_text("step %d / %d\ntasks  r0:%d  r1:%d  r2:%d"%(f,nframes,counts[0],counts[1],counts[2]))
        return dots+trails+goals+[txt]
    step_skip=max(1,nframes//400)
    ani=animation.FuncAnimation(fig,upd,frames=range(0,nframes,step_skip),blit=True,interval=40)
    fn=fname or ("anim_container_seed%s.gif"%seed)
    ani.save(_out(fn),writer=animation.PillowWriter(fps=25))
    plt.close()
    print("saved",fn,"(%d steps, %s)"%(res["steps"],res["stop_reason"]))

if __name__=="__main__":
    grid_figure([1,42,5,88], max_steps=5000)


def single_traj(seed, max_steps=5000, fname=None):
    os.environ["SEED"]=str(seed); os.environ["MAX_STEPS"]=str(max_steps)
    res=cs.run_sim()
    fig,ax=plt.subplots(figsize=(3.5,8.5))
    reason={"all_locked":"DEADLOCK","collision":"COLLISION","max_steps":"survived"}[res["stop_reason"]]
    traj_panel(ax,res,"seed %s — %d steps, %s, tasks=%d"%(seed,res["steps"],reason,sum(res["tasks_done"])))
    plt.tight_layout()
    fn=fname or ("traj_container_seed%s.png"%seed)
    plt.savefig(_out(fn),dpi=95,bbox_inches="tight"); plt.close()
    print("saved",fn)
    return res


def coord_compare(seeds, max_steps=5000, fname="coord_compare.png"):
    """Side-by-side baseline (no coord) vs COORD for each seed."""
    n=len(seeds)
    fig, axes = plt.subplots(2, n, figsize=(2.4*n, 16))
    # Normalize axes to a 2-D (2 x n) indexable form. With n==1, subplots returns
    # a 1-D array of shape (2,), so axes[row][col] raises TypeError; reshape fixes
    # the single-seed case (the bug noted in CODE_REVIEW). (viz #1)
    axes = np.atleast_2d(axes)
    if axes.shape == (1, 2):        # n==1 gives shape (2,) -> atleast_2d -> (1,2)
        axes = axes.reshape(2, 1)
    prev_coord = os.environ.get("COORD")
    for col, s in enumerate(seeds):
        for row, coord in enumerate([False, True]):
            os.environ["SEED"]=str(s); os.environ["MAX_STEPS"]=str(max_steps)
            os.environ["STEER_MODE"]="arch1"; os.environ["COORD"]="1" if coord else "0"
            import importlib, coordination; importlib.reload(coordination); importlib.reload(cs)
            res=cs.run_sim()
            reason={"all_locked":"DEADLOCK","collision":"COLLISION","max_steps":"survived"}[res["stop_reason"]]
            tag=("COORD" if coord else "baseline")
            traj_panel(axes[row][col], res, "seed %s — %s\n%d steps, %s, tasks=%d"%(s,tag,res["steps"],reason,sum(res["tasks_done"])))
    # Restore the COORD env var so callers aren't surprised by the reload side effect.
    if prev_coord is None:
        os.environ.pop("COORD", None)
    else:
        os.environ["COORD"]=prev_coord
    import importlib, coordination; importlib.reload(coordination); importlib.reload(cs)
    fig.suptitle("Coordination layer: baseline (top) vs COORD=1 (bottom)\n"
                 "dynamic jam detection + shared-path retreat token", fontsize=11)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(_out(fname), dpi=85, bbox_inches="tight")
    print("saved", fname)

def coord_animation(seed, max_steps=5000, fname=None):
    os.environ["SEED"]=str(seed); os.environ["MAX_STEPS"]=str(max_steps)
    os.environ["STEER_MODE"]="arch1"; os.environ["COORD"]="1"
    import importlib, coordination; importlib.reload(coordination); importlib.reload(cs)
    make_animation(seed, max_steps=max_steps, fname=fname or ("anim_coord_seed%s.gif"%seed))
