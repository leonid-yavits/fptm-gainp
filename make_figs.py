MERGE_PX = 8


"""Test accuracy vs GainP energy per sample (Mapping S, system model), mean and
standard deviation over seeds. Legend in a dedicated GridSpec row, never
fig.legend (author preference). Labels are placed greedily in free space, and a
label is accepted only if its own marker is the nearest marker."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = os.environ.get("RESULTS", "results")
A = json.load(open(f"{ROOT}/hw_eval_agg.json"))
NAMES = {"wos": "WOS-46985", "alerce": "ALeRCE", "nslkdd": "NSL-KDD"}
ORDER = ["wos", "alerce", "nslkdd"]
STY = {"FPTM": ("#2a78d6", "o"), r"FPTM $\LF{=}1$": ("#eb6834", "s"), "Vanilla TM": ("#1baf7a", "^")}
LEG = {"FPTM": "FPTM", r"FPTM $\LF{=}1$": r"Strict (FPTM, $\mathit{LF}=1$)", "Vanilla TM": "Vanilla TM"}
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})


# ALeRCE FPTM(20) and FPTM(10) markers nearly coincide: 20 labelled up-left, 10 up-right
# NSL-KDD FPTM(40) sits just above the strict(600) marker: labelled straight up
FIXED = {("alerce", "20"): (-10, 10), ("alerce", "10"): (10, 12), ("nslkdd", "40"): (-3, 16)}

def panel_figure(xkey, xlabel, fname):
    fig = plt.figure(figsize=(5.4, 8.0))
    gs = GridSpec(4, 1, height_ratios=[0.06, 0.85, 1.45, 1.55], hspace=0.45, figure=fig)
    lax = fig.add_subplot(gs[0, 0]); lax.axis("off")
    for k, ds in enumerate(ORDER):
        ax = fig.add_subplot(gs[k + 1, 0])
        rs = [a for a in A if a["dataset"] == ds]
        for fam, (c, m) in STY.items():
            fr = sorted([a for a in rs if a["family"] == fam], key=lambda a: a["clauses_per_class"])
            if not fr:
                continue
            x = [a[xkey] for a in fr]; y = [100 * a["test_acc"] for a in fr]
            ax.plot(x, y, color=c, lw=1.0, alpha=0.5, zorder=2)
            xe = [a.get(xkey + "_std") or 0 for a in fr]
            ye = [100 * (a.get("test_acc_std") or 0) for a in fr]
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=c, elinewidth=1, capsize=2, zorder=3)
            ax.scatter(x, y, s=28, c=c, marker=m, edgecolors="#fcfcfb", linewidths=1.4, zorder=5 if fam == "FPTM" else 4)
        ax.set_xscale("log")
        ax.set_title(NAMES[ds], fontsize=9.5, pad=3)
        if k == len(ORDER) - 1:
            ax.set_xlabel(xlabel)
        ax.set_ylabel("Test acc. (%)")
        ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ylo = [100 * (a["test_acc"] - (a.get("test_acc_std") or 0)) for a in rs]
        yhi = [100 * (a["test_acc"] + (a.get("test_acc_std") or 0)) for a in rs]
        lo, hi = min(ylo), max(yhi); pad = max(0.12 * (hi - lo), 0.5)
        ax.set_ylim(lo - pad, hi + 2.2 * pad)
        xs = [a[xkey] for a in rs]
        ax.set_xlim(min(xs) / 1.8, max(xs) * 1.8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, p: f"{v:g}" if f"{v:g}".lstrip("0.")[:1] in ("2", "5") else ""))
        ax.tick_params(axis="x", which="minor", labelsize=8, pad=5.0)  # same baseline as major (tick 3.5 vs 2 pt)
        ax.tick_params(axis="both", which="major", labelsize=8)
        # labels: clauses per class
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        pts = [ax.transData.transform((a[xkey], 100 * a["test_acc"])) for a in rs]
        # sample the family polylines in display space, so labels avoid lines too
        segpts = []
        polysegs = []
        for fam in STY:
            fr = sorted([a for a in rs if a["family"] == fam], key=lambda a: a["clauses_per_class"])
            q = [ax.transData.transform((a[xkey], 100 * a["test_acc"])) for a in fr]
            for (x0, y0), (x1, y1) in zip(q[:-1], q[1:]):
                polysegs.append(((x0, y0), (x1, y1)))
                for u in np.linspace(0, 1, 40):
                    segpts.append((x0 + u * (x1 - x0), y0 + u * (y1 - y0)))
        placed = []
        leaders = []
        def _cross(p1, p2, p3, p4):
            def o(a, b, c):
                return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            return (o(p1, p2, p3) * o(p1, p2, p4) < 0) and (o(p3, p4, p1) * o(p3, p4, p2) < 0)
        cands = [(8, 6), (8, -16), (-10, 8), (-10, -18), (14, 18), (14, -28), (-16, 20),
                 (-16, -30), (24, -4), (-30, -4), (4, 26), (4, -36), (20, 34), (20, -44),
                 (-20, 36), (-20, -46), (36, 10), (-40, 10), (36, -22), (-40, -22),
                 (30, 40), (-34, 40), (30, -50), (-34, -50), (50, 0), (-54, 0),
                 (8, 50), (8, -60), (-12, 56), (-12, -64), (60, 24), (-64, 24), (60, -30), (-64, -30)]
        import math
        cands += [(int(r * math.cos(t)), int(r * math.sin(t))) for r in (22, 34, 48, 64, 80, 100, 120)
                  for t in [k * math.pi / 12 for k in range(24)]]
        cands = sorted(dict.fromkeys(cands), key=lambda c: c[0] ** 2 + c[1] ** 2)
        def try_place(item, strict):
            a, px, py, txt, mem = item
            if (ds, txt) in FIXED:   # hand-checked positions for the tightest pair
                dx, dy = FIXED[(ds, txt)]
                if strict < 2:
                    return False
                t = ax.annotate(txt, (a[xkey], 100 * a["test_acc"]), textcoords="offset points",
                                xytext=(dx, dy), fontsize=8, color=STY[a["family"]][0], fontweight="bold",
                                ha="left" if dx > 0 else "right",
                                arrowprops=dict(arrowstyle="-", color="#a8a79f", lw=0.6, shrinkA=0, shrinkB=4))
                bb = t.get_window_extent(rend)
                placed.append(bb.expanded(1.15, 1.3))
                leaders.append(((px, py), ((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2)))
                return True
            col = STY[a["family"]][0]
            opts = [q for i, q in enumerate(pts) if i not in mem]
            for dx, dy in cands:
                t = ax.annotate(txt, (a[xkey], 100 * a["test_acc"]), textcoords="offset points",
                                xytext=(dx, dy), fontsize=8, color="#52514e", fontweight="bold",
                                ha="left" if dx > 0 else "right")
                bb = t.get_window_extent(rend)
                t.remove()
                cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
                dists = [np.hypot(cx - qx, cy - qy) for qx, qy in opts] or [1e9]
                own = np.hypot(cx - px, cy - py)
                bbx = bb.expanded(1.15, 1.3)
                inside_ax = ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)
                seg = ((px, py), (cx, cy))
                # leader must not pass close to another marker (would suggest the wrong point)
                def _pd(q):
                    vx, vy = cx - px, cy - py
                    L2 = vx * vx + vy * vy
                    u = 0 if L2 == 0 else max(0, min(1, ((q[0] - px) * vx + (q[1] - py) * vy) / L2))
                    return np.hypot(q[0] - px - u * vx, q[1] - py - u * vy)
                def bbdist():   # closest other marker to the label box
                    return min([np.hypot(max(bb.x0 - qx, 0, qx - bb.x1), max(bb.y0 - qy, 0, qy - bb.y1))
                                for qx, qy in opts] or [1e9])
                segs = [(px + u * (cx - px), py + u * (cy - py)) for u in np.linspace(0.25, 1, 30)]
                over_label = any(b.expanded(1.0, 1.0).padded(2).contains(x, y) for b in placed for x, y in segs)
                near_other = any(_pd(q) < 9 for q in opts if np.hypot(q[0] - px, q[1] - py) > 10)
                hit = (any(bbx.overlaps(b) for b in placed)
                       or any(_cross(seg[0], seg[1], l0, l1) for l0, l1 in leaders)
                       or (strict >= 1 and any(_cross(seg[0], seg[1], l0, l1) for l0, l1 in polysegs))
                       or any(bbx.contains(*l0) or bbx.contains(*l1) for l0, l1 in leaders)
                       or any(bbx.contains(qx, qy) for qx, qy in pts)
                       or any(bb.contains(qx, qy) for qx, qy in segpts)
                       or near_other or over_label
                       or (strict >= 2 and own > min(dists) + 6)
                       or (strict == 1 and own > min(dists) + 1)
                       or (strict == 0 and own > min(dists) and bbdist() < 14) or not inside_ax)
                if not hit:
                    placed.append(bbx)
                    leaders.append(seg)
                    ax.annotate(txt, (a[xkey], 100 * a["test_acc"]), textcoords="offset points",
                                xytext=(dx, dy), fontsize=8, color=col, fontweight="bold",
                                ha="left" if dx > 0 else "right",
                                arrowprops=dict(arrowstyle="-", color="#a8a79f", lw=0.6,
                                                shrinkA=0, shrinkB=4))
                    return True
            return False
        # same-family markers closer than 8 px share one label, e.g. "40, 100"
        items, used = [], set()
        for i in sorted(range(len(rs)), key=lambda i: rs[i]["clauses_per_class"]):
            if i in used:
                continue
            mem = {i} | {j for j in range(len(rs)) if j not in used and j != i
                         and rs[j]["family"] == rs[i]["family"]
                         and np.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]) < MERGE_PX}
            used |= mem
            txt = ", ".join(f"{rs[j]['clauses_per_class']:,.0f}"
                            for j in sorted(mem, key=lambda j: rs[j]["clauses_per_class"]))
            if len(mem) > 1:
                print("merged label", ds, txt)
            items.append((rs[i], pts[i][0], pts[i][1], txt, mem))
        items.sort(key=lambda t: (0 if (ds, t[3]) in FIXED else 1, t[1]))
        pending = [it for it in items if not try_place(it, 2)]
        pending2 = [it for it in pending if not try_place(it, 1)]
        for it in pending2:
            print(("relaxed label" if try_place(it, 0) else "UNPLACED label"), ds, it[3])
    handles = [Line2D([], [], linestyle="-", lw=1.0, marker=m, color=c, markersize=6, label=LEG[k])
               for k, (c, m) in STY.items()]
    lax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=8.5, handlelength=2.2)
    fig.savefig(f"paper/figures/{fname}.pdf", bbox_inches="tight")
    fig.savefig(f"paper/figures/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


os.makedirs("paper/figures", exist_ok=True)
for f in ("acc_vs_cycles_S.pdf", "acc_vs_cycles_S.png"):
    if os.path.exists(f"paper/figures/{f}"):
        os.remove(f"paper/figures/{f}")
panel_figure("S_n7o_energy_nJ_per_sample", "Energy per sample (nJ)", "acc_vs_energy_S")
print("figures written")
