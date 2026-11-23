"""Fig. 4: test accuracy versus predicted Mapping S energy (optimistic 7 nm corner) for the
accuracy-oriented hyperparameters (main experiments, filled markers) and the energy-oriented
hyperparameters (hollow markers), with the joint accuracy/energy front. Reads results/retuned_agg.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

R = json.load(open("results/retuned_agg.json"))
EK = "S_n7o_energy_nJ_per_sample"
COL = {"FPTM": "#2a78d6", "Strict": "#eb6834", "Vanilla": "#1baf7a"}
MRK = {"FPTM": "o", "Strict": "s", "Vanilla": "^"}
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
# label offsets (points) for the front members, checked by eye on the rendered figure
OFF = {("alerce", "E:FPTM-2"): (14, -12), ("alerce", "E:FPTM-20"): (-16, 4), ("alerce", "E:Strict-200"): (-12, 16),
       ("alerce", "E:Strict-400"): (-6, 16), ("alerce", "E:Strict-1000"): (-6, 16),
       ("nslkdd", "E:FPTM-2"): (14, -14), ("nslkdd", "E:FPTM-20"): (6, 16), ("nslkdd", "A:FPTM-40"): (-10, 16)}
NAME = {"FPTM": "FPTM", "Strict": "Strict", "Vanilla": "Vanilla"}
PANEL = [("alerce", "(a) ALeRCE"), ("nslkdd", "(b) NSL-KDD")]

fig = plt.figure(figsize=(5.4, 6.6))
og = GridSpec(2, 1, height_ratios=[0.09, 2.0], hspace=0.14, figure=fig)
gs = og[1].subgridspec(2, 1, hspace=0.42)
for i, (ds, title) in enumerate(PANEL):
    ax = fig.add_subplot(gs[i])
    d = R[ds]
    pts = {}
    for tag, sub in (("A", d["accuracy_oriented"]), ("E", d["energy_oriented"])):
        for k, m in sub.items():
            pts[f"{tag}:{k}"] = m
            f = m["family"]
            ax.scatter(m[EK], 100 * m["test_acc"], s=30, marker=MRK[f], zorder=4,
                       facecolors=COL[f] if tag == "A" else "white", edgecolors=COL[f], linewidths=1.3)
    fr = d["front_opt"]
    xs = [pts[k][EK] for k in fr]; ys = [100 * pts[k]["test_acc"] for k in fr]
    sx, sy = [xs[0]], [ys[0]]
    for x, y in zip(xs[1:], ys[1:]):
        sx += [x, x]; sy += [sy[-1], y]
    ax.plot(sx, sy, color="#52514e", lw=0.9, ls="--", zorder=2)
    for k in fr:
        m = pts[k]
        ax.scatter(m[EK], 100 * m["test_acc"], s=95, marker=MRK[m["family"]], facecolors="none",
                   edgecolors="#0b0b0b", linewidths=0.9, zorder=5)
    for k in fr:
        m = pts[k]; dx, dy = OFF[(ds, k)]
        ax.annotate(f"{NAME[m['family']]} {m['clauses']}", (m[EK], 100 * m["test_acc"]), textcoords="offset points",
                    xytext=(dx, dy), ha="left" if dx > 0 else "right", va="center", fontsize=7.5, color="#0b0b0b",
                    arrowprops=dict(arrowstyle="-", color="#a8a79f", lw=0.6, shrinkA=1, shrinkB=4))
    ax.set_xscale("log")
    ax.set_xlabel("Energy per sample, Mapping S, optimistic corner (nJ)")
    ax.set_ylabel("Test accuracy (%)")
    yf = [100 * pts[k]["test_acc"] for k in fr]
    lo, hi = min(yf) - 1.3, max(yf) + 0.7          # zoom on the front; points below are listed in the caption
    ax.set_ylim(lo, hi)
    below = sorted(f"{k} {100*m['test_acc']:.2f}" for k, m in pts.items() if 100 * m["test_acc"] < lo)
    print(ds, "below axis:", below)
    xs_all = [m[EK] for m in pts.values()]
    ax.set_xlim(min(xs_all) / 1.6, max(xs_all) * 1.6)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(True, which="major", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
lax = fig.add_subplot(og[0]); lax.axis("off")
H = [Line2D([], [], ls="", marker=MRK[f], mfc=COL[f], mec=COL[f], label=f"{NAME[f]}") for f in ("FPTM", "Strict", "Vanilla")]
H += [Line2D([], [], ls="", marker="o", mfc="white", mec="#52514e", label="energy-oriented (hollow)"),
      Line2D([], [], color="#52514e", ls="--", lw=0.9, label="accuracy/energy front"),
      Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec="#0b0b0b", mew=0.9, label="front member (ringed)")]
lax.legend(handles=H, loc="center", ncol=3, frameon=False, fontsize=7.5, handletextpad=0.3, columnspacing=1.0)
fig.savefig("paper/figures/acc_energy_front.pdf", bbox_inches="tight")
fig.savefig("paper/figures/acc_energy_front.png", dpi=150, bbox_inches="tight")
print("ok")
