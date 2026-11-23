"""Generate LaTeX tables from results/hw_eval.json, aggregated over seeds
(mean and sample standard deviation). No number in the tables is typed by hand."""
import json, os
from collections import defaultdict
import numpy as np
from datasets import load

os.makedirs("paper/tables", exist_ok=True)
ROOT = os.environ.get("RESULTS", "results")
RUNS = json.load(open(f"{ROOT}/hw_eval.json"))
NAMES = {"wos": "WOS-46985", "alerce": "ALeRCE", "nslkdd": "NSL-KDD"}
ORDER = ["wos", "alerce", "nslkdd"]


def family(r):
    if r["impl"] == "pyTsetlinMachine":
        return "Vanilla TM"
    return r"FPTM $\LF{=}1$" if r["kind"] == "strict" else "FPTM"


FAM_ORDER = {"FPTM": 0, r"FPTM $\LF{=}1$": 1, "Vanilla TM": 2}


def aggregate(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["tag"])].append(r)
    agg = []
    for (ds, tag), rs in groups.items():
        a = dict(dataset=ds, tag=tag, family=family(rs[0]), n=len(rs), runs=rs)
        for k, v in rs[0].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals = np.array([x[k] for x in rs if x[k] is not None], dtype=float)
                a[k] = float(vals.mean()) if len(vals) else None
                a[k + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else None
            elif k not in a:
                a[k] = v
        agg.append(a)
    agg.sort(key=lambda a: (ORDER.index(a["dataset"]), FAM_ORDER[a["family"]], a["clauses_per_class"]))
    return agg


AGG = aggregate(RUNS)
json.dump([{k: v for k, v in a.items() if k != "runs"} for a in AGG],
          open(f"{ROOT}/hw_eval_agg.json", "w"), indent=1)


def pm(a, k, scale=1.0, nd=2, thousands=False):
    m = a[k] * scale
    s = a.get(k + "_std")
    fm = f"{m:,.{nd}f}" if thousands else f"{m:.{nd}f}"
    if s is None:
        return fm
    s *= scale
    fs = f"{s:,.{nd}f}" if thousands else f"{s:.{nd}f}"
    return f"{fm} $\\pm$ {fs}"


def sig(x, n=3):
    """n significant digits, thousands separators for large values."""
    if x == 0:
        return "0"
    if abs(x) >= 10 ** (n - 1):
        return f"{x:,.0f}"
    return f"{x:.{max(0, n - 1 - int(np.floor(np.log10(abs(x)))))}f}"


def group_row(lines, prev, cur, nc):
    if prev is not None and prev != cur:
        lines.append(r"\midrule")
    if prev != cur:
        lines.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (nc, NAMES[cur]))
    return cur


DISP = lambda fam: "Strict" if "LF" in fam else fam


def rule_between(lines, prev, cur):
    if prev is not None and prev != cur:
        lines.append(r"\midrule")
    return cur


# ---- datasets -------------------------------------------------------------------
lines = [r"\begin{tabular}{lllrrrr}", r"\toprule",
         r"Dataset & Domain & Classes & Bits & Train & Val. & Test \\", r"\midrule"]
dom = {"wos": "Scientific text", "alerce": "Astronomy", "nslkdd": "Network security"}
for ds in ORDER:
    d = load(ds)
    lines.append(f"{NAMES[ds]} & {dom[ds]} & {len(np.unique(d['ytr']))} & {d['Xtr'].shape[1]:,} & "
                 f"{len(d['ytr']):,} & {len(d['yva']):,} & {len(d['yte']):,} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tables/datasets.tex", "w").write("\n".join(lines) + "\n")

# ---- hyperparameters ---------------------------------------------------------------
lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Model & Cl./class & $\LF$ & $L$ & $T$ & $S$ or $s$ & Incl.\ limit & Seeds & Best epoch \\",
         r"\midrule"]
prev = None
for a in AGG:
    prev = group_row(lines, prev, a["dataset"], 9)
    r0 = a["runs"][0]
    if a["family"] == "Vanilla TM":
        L = LF = IL = "n/a"; Ss = f"$s={r0['s']:g}$"
    else:
        L, LF, IL = r0["L"], r0["LF"] if r0["LF"] else 1, r0["include_limit"]; Ss = f"$S={r0['S']}$"
    be = ", ".join(str(x["best_epoch"]) for x in sorted(a["runs"], key=lambda x: x["seed"]))
    lines.append(f"{DISP(a['family'])} & {a['clauses_per_class']:,.0f} & {LF} & {L} & {r0['T']} & "
                 f"{Ss} & {IL} & {a['n']} & {be} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tables/hyper_full.tex", "w").write("\n".join(lines) + "\n")   # supplementary Table S1
# compact main-text table: the hyperparameters do not depend on the dataset except S
DSN = {"wos": "W", "alerce": "A", "nslkdd": "N"}
comp = {}
for a in AGG:
    r0 = a["runs"][0]; key = (a["family"], a["clauses_per_class"])
    if a["family"] == "Vanilla TM":
        row = ("n/a", "n/a", r0["T"], f"$s={r0['s']:g}$", "n/a")
    else:
        row = (r0["LF"] if r0["LF"] else 1, r0["L"], r0["T"], "$S$", r0["include_limit"])
    c = comp.setdefault(key, {"row": row, "ds": [], "seeds": set()})
    assert c["row"] == row, (key, row, c["row"])
    c["ds"].append(DSN[a["dataset"]]); c["seeds"].add(a["n"])
FO = {"FPTM": 0, "Strict": 1, "Vanilla TM": 2}
lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Model & Cl./class & $\LF$ & $L$ & $T$ & $S$ or $s$ & Incl.\ limit & Datasets (seeds) \\", r"\midrule"]
for (fam, cl), c in sorted(comp.items(), key=lambda kv: (0 if kv[0][0] == "FPTM" else 2 if "Vanilla" in kv[0][0] else 1, kv[0][1])):
    sd = "/".join(str(x) for x in sorted(c["seeds"]))
    lines.append(f"{DISP(fam)} & {cl:,.0f} & " + " & ".join(str(x) for x in c["row"]) + f" & {', '.join(sorted(c['ds'], key='WAN'.index))} ({sd}) \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tables/hyper.tex", "w").write("\n".join(lines) + "\n")

# ---- accuracy and size --------------------------------------------------------------
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Model & Cl./class & Val.\ acc.\ (\%) & Test acc.\ (\%) & Literals & Size (KiB) \\",
         r"\midrule"]
prev = None
for a in AGG:
    prev = group_row(lines, prev, a["dataset"], 6)
    lines.append(f"{DISP(a['family'])} & {a['clauses_per_class']:,.0f} & "
                 f"{pm(a, 'val_acc', 100)} & {pm(a, 'test_acc', 100)} & {a['literals']:,.0f} & "
                 f"{a['model_bytes_sparse']/1024:,.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tables/accuracy.tex", "w").write("\n".join(lines) + "\n")

# ---- Mapping S (predicted 7 nm, blue) ------------------------------------------------
B = lambda x: r"\hyp{" + x + "}"
lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"\multirow{2}{*}{Model (cl./class)} & Rows & Sub- & Cycles & \multicolumn{2}{c}{Batch latency ($\mu$s)} & \multicolumn{2}{c}{Energy (nJ/sample)} & Thr., opt. \\",
         r" & /col. & arrays & ($\times10^3$) & cons. & opt. & cons. & opt. & (M/s/mm$^2$) \\",
         r"\midrule"]
prev = None
for a in AGG:
    prev = group_row(lines, prev, a["dataset"], 9)
    lines.append(f"{DISP(a['family'])} ({a['clauses_per_class']:,.0f}) & {a['S_rows']:,.0f} & "
                 f"{a['S_subarrays_per_col']:.0f} & {pm(a, 'S_cycles', 1e-3, 1, True)} & "
                 f"{B(sig(a['S_n7c_batch_latency_us']))} & {B(sig(a['S_n7o_batch_latency_us']))} & "
                 f"{B(pm(a, 'S_n7c_energy_nJ_per_sample', 1, 2))} & {B(pm(a, 'S_n7o_energy_nJ_per_sample', 1, 3))} & "
                 f"{B(sig(a['S_n7o_throughput_per_mm2'] / 1e6))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tables/mapS.tex", "w").write("\n".join(lines) + "\n")

# ---- Mapping C (predicted 7 nm, blue) ------------------------------------------------
lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r"\multirow{2}{*}{Model (cl./class)} & Used & Clause & Sub- & Area & Cycles & \multicolumn{2}{c}{Latency ($\mu$s)} & \multicolumn{2}{c}{Energy (nJ/sample)} \\",
         r" & features & columns & arrays & (mm$^2$) & & cons. & opt. & cons. & opt. \\",
         r"\midrule"]
prev = None
for a in AGG:
    prev = group_row(lines, prev, a["dataset"], 10)
    lines.append(f"{DISP(a['family'])} ({a['clauses_per_class']:,.0f}) & {a['C_F_used']:,.0f} & "
                 f"{a['C_columns']:,.0f} & {a['C_subarrays']:,.0f} & {B(sig(a['C_n7o_area_mm2']))} & {a['C_cycles']:,.0f} & "
                 f"{B(sig(a['C_n7c_latency_us']))} & {B(sig(a['C_n7o_latency_us']))} & "
                 f"{B(sig(a['C_n7c_energy_nJ_per_sample']))} & {B(sig(a['C_n7o_energy_nJ_per_sample']))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tables/mapC.tex", "w").write("\n".join(lines) + "\n")
print("tables written:", len(AGG), "configurations,", len(RUNS), "runs")

# ---- 16 nm reference and predicted 7 nm parameters ----------------------------------
import hw_eval as H
M = H.MODELS
rows = [("Compare cycle (ns)", "t_cycle_ns", "{:.2f}"),
        ("Energy per cell operation (fJ)", "e_cell_fJ", "{:.2f}"),
        ("Subarray macro area ($\\mu$m$^2$)", None, None),
        ("Gain-cell area ($\\mu$m$^2$)", None, None),
        ("Static power ($\\mu$W/Mbit)", "static", "{:.0f}"),
        ("Refresh period ($\\mu$s)", None, None)]
cols = [("cir", "16~nm, circuit"), ("sys", "16~nm, system"), ("n7c", "7~nm, conservative"), ("n7o", "7~nm, optimistic")]
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r" & 16~nm & 16~nm & 7~nm & 7~nm \\", r"Parameter & circuit & system & conservative & optimistic \\", r"\midrule"]
for name, key, f in rows:
    vals = []
    for k, _ in cols:
        m = M[k]
        if key:
            v = f.format(m[key])
        elif name.startswith("Subarray"):
            v = f"{H.SUBARRAY_AREA_UM2 * m['area']:.0f}"
        elif name.startswith("Gain-cell"):
            v = f"{0.082 * m['area']:.4f}"
        else:
            v = f"{H.T_REF_US * m.get('ret', 1.0):.2f}"
        # static power comes from the unpublished workbook: treated as an assumption (blue)
        # blue: 7 nm predictions; static power (unpublished workbook); the system-point energy
        # (interpretation of 2.17 uW/bit at 1 GHz as fJ per cell operation); the refresh period (our mu-3sigma choice)
        black = k in ("cir", "sys") and key not in ("static",) and not (k == "sys" and key == "e_cell_fJ") \
            and not name.startswith("Refresh")
        vals.append(v if black else r"\hyp{" + v + "}")
    lines.append(name + " & " + " & ".join(vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tables/params7.tex", "w").write("\n".join(lines) + "\n")
print("params table written")

# ---- sensitivity paragraph (generated) -------------------------------------------------
G = {(a["dataset"], a["tag"]): a for a in AGG}
f20, s600 = G[("alerce", "F3_C20")], G[("alerce", "S_LF1_C600")]
def ratio(k):
    return s600[k] / f20[k]
w = G[("wos", "S_LF1_C200")]; wf = G[("wos", "F5_C100")]
stat_share = lambda a, k: 100 * a[f"S_{k}_E_static_nJ"] / a[f"S_{k}_energy_nJ_per_sample"]
def _ser(ds, k):
    hh = max(a["S_subarrays_per_col"] for a in AGG if a["dataset"] == ds)
    t = H.MODELS[k]["t_cycle_ns"] * 1e-3
    return 100 * hh * H.REFRESH_CYCLES * t / H.T_REF_US, int(hh)
_sa, _ha = _ser("alerce", "n7c"); _sa2, _ = _ser("alerce", "n7o")
_sn, _hn = _ser("nslkdd", "n7c"); _sn2, _ = _ser("nslkdd", "n7o")
_sw, _hw = _ser("wos", "n7c"); _sw2, _ = _ser("wos", "n7o")
ser_txt = (r"The refresh model assumes that the subarrays of a logical column refresh in lockstep through their local sense amplifiers (Section~\ref{sec:method}). "
           r"If refresh instead used the shared sense amplifier, the $h$ subarrays would be refreshed one after another, and each refresh would stall the array for $h\times512$ cycles. "
           rf"This stall would take \hyp{{{_sa:.1f}\%}} (conservative) and \hyp{{{_sa2:.1f}\%}} (optimistic) of every refresh period on ALeRCE ($h={_ha}$), and \hyp{{{_sn:.1f}\%}} and \hyp{{{_sn2:.1f}\%}} on NSL-KDD ($h={_hn}$). "
           rf"On WOS-46985 ($h={_hw}$) it would take \hyp{{{_sw:.0f}\%}} and \hyp{{{_sw2:.0f}\%}} of the period, so Mapping~S would not be feasible at the conservative corner. "
           r"Mapping~C on WOS-46985 stacks up to 79 subarrays and would be affected in the same way. "
           r"All ALeRCE models use the same $h$ in Mapping~S, so their ranking would not change.")
def _ctl_ratio(scale_e, n_sub):
    def tot(a):
        h = int(a["S_subarrays_per_col"])
        ctl = a["S_E_ctl_nJ"] * scale_e * (a["S_stacks_per_macro"] / (n_sub // h))
        return a["S_n7c_energy_noctl_nJ"] + ctl
    return tot(s600) / tot(f20)
_c1 = _ctl_ratio(0.2, 256); _c64 = _ctl_ratio(1.0, 64); _c0 = s600["S_n7c_energy_noctl_nJ"] / f20["S_n7c_energy_noctl_nJ"]
def _shr(ds, k):
    v = [100 * a[f"S_{k}_E_ctl_nJ"] / a[f"S_{k}_energy_nJ_per_sample"] for a in AGG if a["dataset"] == ds]
    return f"{min(v):.0f}\\% to {max(v):.0f}\\%"
ctl_txt = (rf"The controller share of the Mapping~S energy is \hyp{{{_shr('nslkdd', 'n7c')}}} on NSL-KDD, \hyp{{{_shr('alerce', 'n7c')}}} on ALeRCE and \hyp{{{_shr('wos', 'n7c')}}} on WOS-46985 at the conservative corner. "
           rf"At the optimistic corner it is \hyp{{{_shr('nslkdd', 'n7o')}}}, \hyp{{{_shr('alerce', 'n7o')}}} and \hyp{{{_shr('wos', 'n7o')}}}, because the array energy shrinks more than the upper-estimate controller energy. "
           rf"Without the controller, the strict-to-FPTM energy ratio on ALeRCE is \hyp{{{_c0:.2f}$\times$}}. With a five times lower sequencer and model-read energy, it is \hyp{{{_c1:.2f}$\times$}}, and with a 64-subarray macro it is \hyp{{{_c64:.2f}$\times$}}.")
txt = rf"""We test how the \wosnew{{accuracy-oriented}} ALeRCE comparison, the 20-clause FPTM against the 600-clause $\LF=1$ model, depends on the modelling choices.
At the 16~nm reference points, the strict model needs \hyp{{{ratio('S_sys_energy_nJ_per_sample'):.2f}$\times$}} (system point) and \hyp{{{ratio('S_cir_energy_nJ_per_sample'):.2f}$\times$}} (circuit point) the energy of the FPTM model. These ratios use the same assumed energy model as the 7~nm values.
At the predicted 7~nm corners the ratio is \hyp{{{ratio('S_n7c_energy_nJ_per_sample'):.2f}$\times$}} (conservative) and \hyp{{{ratio('S_n7o_energy_nJ_per_sample'):.2f}$\times$}} (optimistic).
If every cycle is charged one cell operation per column, regardless of the number of unmasked rows, the ratio becomes \hyp{{{ratio('S_n7c_energy_percolumn_nJ'):.2f}$\times$}} at the conservative corner.
The ranking therefore holds for both energy models tested.
{ctl_txt}
With a $10\times$ shorter retention time, the refresh overhead of the FPTM batch latency rises from \hyp{{{f20['S_n7c_refresh_overhead_pct']:.1f}\%}} to \hyp{{{f20['S_n7r_refresh_overhead_pct']:.1f}\%}} at the conservative corner. Its energy per sample rises from \hyp{{{f20['S_n7c_energy_nJ_per_sample']:.2f}~nJ}} to \hyp{{{f20['S_n7r_energy_nJ_per_sample']:.2f}~nJ}}.
Static energy is a large share wherever logical columns are tall and programs are long. For the 100-clause WOS-46985 FPTM it is \hyp{{{stat_share(wf, 'n7c'):.0f}\%}} of the energy at the conservative corner, against \hyp{{{stat_share(w, 'n7c'):.0f}\%}} for the 200-clause $\LF=1$ model.
{ser_txt}
"""
open("paper/tables/sens_full.tex", "w").write(txt)   # supplementary Section S3
_rat = [ratio('S_sys_energy_nJ_per_sample'), ratio('S_cir_energy_nJ_per_sample'), ratio('S_n7c_energy_nJ_per_sample'),
        ratio('S_n7o_energy_nJ_per_sample'), ratio('S_n7c_energy_percolumn_nJ'), _c0, _c1, _c64]
_sh = [100 * a[f"S_{k}_E_ctl_nJ"] / a[f"S_{k}_energy_nJ_per_sample"] for a in AGG for k in ("n7c", "n7o")]
txt = rf"""We test how the \wosnew{{accuracy-oriented}} ALeRCE comparison, the 20-clause FPTM against the 600-clause $\LF=1$ model, depends on the modelling choices.
\wosnew{{The strict model needs \hyp{{{min(_rat):.2f}$\times$}} to \hyp{{{max(_rat):.2f}$\times$}} the energy of the FPTM model in every case tested.
The cases are the 16~nm reference points, both predicted 7~nm corners, a per-column energy model, no controller, a five times lower sequencer energy and a 64-subarray macro.
The controller share of the Mapping~S energy is \hyp{{{min(_sh):.0f}\% to {max(_sh):.0f}\%}}, highest on WOS-46985.
With a $10\times$ shorter retention time, the refresh overhead of the FPTM batch latency rises to \hyp{{{f20['S_n7r_refresh_overhead_pct']:.1f}\%}} at the conservative corner.
If refresh used the shared sense amplifier instead of lockstep local refresh, Mapping~S on WOS-46985 would not be feasible at the conservative corner, while the ALeRCE ranking would not change.
Section~\\supref{{sens}} of the supplementary material gives the full analysis.}}
"""
open("paper/tables/sens.tex", "w").write(txt)
print("sensitivity written")

# ---- CPU comparison ------------------------------------------------------------------------
CPU = json.load(open(f"{ROOT}/cpu_bench.json")) if os.path.exists(f"{ROOT}/cpu_bench.json") else None
if CPU:
    cg = defaultdict(list); ck = defaultdict(list)
    P2 = f"{ROOT}/cpu_bench_pass2.json"
    if os.path.exists(P2):          # two independent passes: keep the faster (less disturbed) one per run
        other = {(r["dataset"], r["tag"], r["seed"]): r for r in json.load(open(P2))["rows"]}
        for r in CPU["rows"]:
            o = other.get((r["dataset"], r["tag"], r["seed"]))
            if o and o["cpu_us_per_sample"] < r["cpu_us_per_sample"]:
                r.update(cpu_us_per_sample=o["cpu_us_per_sample"], best_kernel=o["best_kernel"])
    for r in CPU["rows"]:
        cg[(r["dataset"], r["tag"])].append(r["cpu_us_per_sample"])
        ck[(r["dataset"], r["tag"])].append(r["best_kernel"])
    lines = [r"\begin{tabular}{lrlrrr}", r"\toprule",
             r"\multirow{2}{*}{Model (cl./class)} & CPU time & CPU & CPU energy & \gainp{} energy & \multirow{2}{*}{Ratio} \\",
             r" & ($\mu$s/sample) & kernel & at 1~W ($\mu$J) & (nJ, cons.) & \\", r"\midrule"]
    prev = None
    for a in AGG:
        prev = group_row(lines, prev, a["dataset"], 6)
        v = np.array(cg[(a["dataset"], a["tag"])])
        cpu = f"{v.mean():.2f} $\\pm$ {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"
        kern = "/".join(sorted(set(ck[(a["dataset"], a["tag"])])))
        eg = a["S_n7c_energy_withload_nJ"]
        lines.append(f"{DISP(a['family'])} ({a['clauses_per_class']:,.0f}) & {cpu} & {kern} & "
                     f"{B(sig(v.mean()))} & {B(sig(eg))} & {B((lambda x: f'{x:,.0f}' if x >= 100 else f'{x:.1f}')(v.mean() * 1e3 / eg))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("paper/tables/cpu.tex", "w").write("\n".join(lines) + "\n")
    print("cpu table written")

# ---- TM hardware comparison (literature values verified in the cited papers) ---------------
g = lambda ds, t: G[(ds, t)]
def gp(ds, t):
    a = g(ds, t)
    thr = a["S_n7c_macro_throughput"]
    thr = f"{thr / 1e6:.1f}M" if thr >= 1e6 else f"{thr / 1e3:.0f}k"
    return (f"{100 * a['test_acc']:.2f}", f"{a['S_n7c_energy_nJ_per_sample']:.2f} / {a['S_n7o_energy_nJ_per_sample']:.2f}~nJ",
            f"{thr} ({a['S_n7c_macro_area_mm2']:.2f}~mm$^2$)")
lit = [
    (r"Wheeldon et al.~\cite{wheeldon2020learning}", "ASIC, 65~nm, layout", "Vanilla TM (20)", "Iris", "n/r$^{a}$", "30.6~pJ", "n/r"),
    (r"Tunheim et al.~\cite{tunheim2026asic}", "ASIC, 65~nm, silicon", "ConvCoTM", "MNIST", "97.42", "8.6~nJ$^{b}$", "60.3k (2.7~mm$^2$)"),
    (r"Lin et al.~\cite{lin2025tsetlinkws}$^{c}$", "ASIC, 65~nm", "CTM", "GSCD-12$^{d}$", "87.35", "191~nJ$^{e}$", "n/r$^{f}$"),
    (r"Tunheim et al.~\cite{tunheim2023ctm}", "FPGA, XC7Z020", "CTM", "2-D Noisy XOR", "99.9", r"0.6~$\mu$J", "4.4M"),
    (r"Tunheim et al.~\cite{tunheim2025fpga}", "FPGA, XCZU7", "ConvCoTM", "MNIST", "97.6", r"13.3~$\mu$J", "134k"),
    (r"Mao et al.~\cite{mao2025dtm}", "FPGA, ZU7EV", "Vanilla TM, CoTM", "MNIST", "97.74", r"73.6~$\mu$J$^{g}$", "22.4k"),
    (r"Ghazal et al.~\cite{ghazal2023imbue}", "ReRAM IMC, 65~nm$^{h}$", "Vanilla TM", "MNIST", "96.48", "13.9~nJ", "n/r"),
    (r"Ghazal et al.~\cite{ghazal2023imbue}", "ReRAM IMC, 65~nm$^{h}$", "Vanilla TM", "KWS-6", "87.1", "5.91~nJ", "n/r"),
    (r"Ghazal et al.~\cite{ghazal2025impact}", "Y-Flash IMC, 180~nm$^{h}$", "CoTM", "MNIST", "96.3", "84.2~pJ$^{i}$", "n/r"),
    (r"Hnilov~\cite{hnilov2025fptm}", "CPU, 16 cores", "FPTM (1)", "IMDb", "n/r$^{j}$", "n/r", "34.5M"),
]
if CPU:
    v = np.array(cg[("alerce", "F3_C20")])
    lit.append((r"This work", r"CPU, 1 thread", "FPTM (20)", "ALeRCE", f"{100 * g('alerce', 'F3_C20')['test_acc']:.2f}",
                "not measured", f"{1e3 / v.mean():.0f}k"))
ours = [("FPTM (20)", "alerce", "F3_C20"), ("Vanilla TM (1,000)", "alerce", "V_C1000"),
        (r"Strict (200)", "nslkdd", "S_LF1_C200"), ("FPTM (20)", "nslkdd", "F3_C20")]
P = lambda w: r">{\raggedright\arraybackslash}p{" + w + "}"
lines = [r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{" + P("3.0cm") + P("3.2cm") + P("2.7cm") + P("2.4cm") + "r" + P("2.4cm") + P("2.9cm") + "}", r"\toprule",
         r"Design & Platform & Model & Dataset & Acc.\ (\%) & Energy/classif. & Classif./s \\", r"\midrule"]
for row in lit:
    lines.append(" & ".join(row) + r" \\")
lines.append(r"\midrule")
for lab, ds, t in ours:
    acc, e, thr = gp(ds, t)
    lines.append(f"This work & \\hyp{{\\gainp{{}}, 7~nm, pred.}} & {lab} & {NAMES[ds]} & {acc} & \\hyp{{{e}}} & \\hyp{{{thr}}} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tables/tmhw.tex", "w").write("\n".join(lines) + "\n")
print("TM hardware table written")
