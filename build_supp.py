"""Build the supplementary material (out/overleaf/supplementary.tex).

Sections follow the order of first citation in the main text. Text moved out of the main text is stored in
paper/supp_moved.json. Main-text numbers (sections, tables, figures, equations, references) are read from the
compiled main.aux, given by the MAIN_AUX environment variable. After the supplement is compiled, run
`python3 build_supp.py --labels <supplementary.aux>` to write out/overleaf/supp_labels.json, which assemble.py
uses to fill the \\supref{...} pointers of the main text. build_all.sh runs both passes."""
import json, math, os, re, sys

OUT = "out/overleaf"
if len(sys.argv) > 2 and sys.argv[1] == "--labels":
    aux = open(sys.argv[2]).read()
    lab = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux))
    keep = {k: v for k, v in lab.items() if v.startswith("S")}
    json.dump(keep, open(os.path.join(OUT, "supp_labels.json"), "w"), indent=1, sort_keys=True)
    print(keep)
    sys.exit(0)

from scipy.stats import spearmanr
AUX = open(os.environ["MAIN_AUX"]).read()
LAB = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", AUX))
CIT = dict(re.findall(r"\\bibcite\{([^}]*)\}\{\{([^}]*)\}", AUX))
INTERNAL = {"tab:cpu", "tab:mapC", "tab:hyperfull", "fig:paretofull", "tab:spearman", "sens", "modelling"}


def _fix(t):
    """Turn main-text references into plain numbers; keep references to items that live in the supplement."""
    t = t.replace(r"Section~\ref{sec:sens}", r"Section~%s of the main text and Section~\ref{sens}" % LAB["sec:sens"])
    t = re.sub(r"of~\\cite\{levi2026gainp\}", r"in the \\gainp{} article (reference~[%s] of the main text)" % CIT["levi2026gainp"], t)
    t = re.sub(r"~?\\cite\{([^}]*)\}", lambda m: " (reference~[%s] of the main text)" % CIT[m.group(1)], t)
    t = re.sub(r"Eq\.~\(\\ref\{([^}]*)\}\)", lambda m: "Eq.~(%s) of the main text" % LAB[m.group(1)], t)
    t = re.sub(r"(Section|Table|Fig\.)~\\ref\{([^}]*)\}",
               lambda m: m.group(0) if m.group(2) in INTERNAL else "%s~%s of the main text" % (m.group(1), LAB[m.group(2)]), t)
    t = t.replace(r"\wosnew{", "{")
    for lab in re.findall(r"\\ref\{([^}]*)\}", t):
        assert lab in INTERNAL, lab
    assert r"\cite{" not in t, t
    return t


MV = json.load(open("paper/supp_moved.json"))
# ---- Spearman table
R = json.load(open("results/retuned_agg.json"))
NAMES = {"alerce": "ALeRCE", "nslkdd": "NSL-KDD", "wos": "WOS-46985"}
rows = []
for ds in ("wos", "alerce", "nslkdd"):
    ms = [m for st in ("accuracy_oriented", "energy_oriented") for m in R[ds][st].values() if m["family"] == "FPTM"]
    cyc = [m["S_cycles"] for m in ms]
    cl = [m["clauses"] for m in ms]
    lit = [m["clauses"] * m["lit_per_clause"] for m in ms]
    wl = [m["clauses"] * m["lit_per_clause"] * math.log2(m["LF"] + 1) for m in ms]
    rows.append((NAMES[ds], len(ms), spearmanr(cyc, cl)[0], spearmanr(cyc, lit)[0], spearmanr(cyc, wl)[0]))
sp = "\n".join(f"{n} & {k} & {a:.2f} & {b:.2f} & {c:.2f} \\\\" for n, k, a, b, c in rows)
lo_w, hi_w = min(r[4] for r in rows), max(r[4] for r in rows)
lo_c, hi_c = min(r[2] for r in rows), max(r[2] for r in rows)

# ---- moved text
sens = open("paper/tables/sens_full.tex").read().replace(r"\wosnew{accuracy-oriented}", "accuracy-oriented")
sens = sens.replace(r"(Section~\ref{sec:method})", r"(Section~\ref{modelling})")
sens = _fix(sens)
hyper = open("paper/tables/hyper_full.tex").read()
TRAIN, EVAL, REFR, LOAD, LEAK, M3, M6, CPUPROT = (_fix(MV[k]) for k in ("TRAIN", "EVAL", "REFR", "LOAD", "LEAK", "M3", "M6", "CPU"))
cpuf = MV["CPUFULL"]
CPUTAB = cpuf[cpuf.index(r"\begin{table}"):cpuf.index(r"\end{table}") + len(r"\end{table}")]
CPUTAB = _fix(CPUTAB).replace(r"\input{tables/cpu}", open("paper/tables/cpu.tex").read())
CPUBOUND = _fix(cpuf[cpuf.index(r"\hyp{We bound the CPU energy"):].strip())
MAPC = _fix(MV["MAPC"]).replace(r"\input{tables/mapC}", open("paper/tables/mapC.tex").read())
FIG = _fix(MV["FIG"].replace(r"\label{fig:pareto}", r"\label{fig:paretofull}")).replace("[p]", "[H]").replace(
    r"height=0.82\textheight", r"height=0.78\textheight")
DISC = _fix(MV["DISC"])
DISC = DISC.replace("\\paragraph{Strict models from the FPTM trainer}\n", "\\textit{Strict models from the FPTM trainer.} ")
DISC = DISC.replace("\\paragraph{Mapping choice}\n", "\\textit{Mapping choice.} ")
DISC = DISC.replace(r"On ALeRCE, the 20-clause FPTM needs about $47\times$", r"On ALeRCE, the accuracy-oriented 20-clause FPTM needs about $47\times$")
assert r"\paragraph" not in DISC

doc = r"""\documentclass[preprint,12pt]{elsarticle}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{rotating}
\usepackage{multirow}
\usepackage{float}
\newcommand{\gainp}{GainP}
\newcommand{\hyp}[1]{#1}   % predicted values
\newcommand{\LF}{\mathit{LF}}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\journal{Expert Systems with Applications}
\begin{document}
\sloppy
\begin{frontmatter}
\title{Supplementary material for: Fuzzy-Pattern Tsetlin Machine Inference on a Gain Cell Embedded DRAM Associative In-Memory Processor}
\author{Artem Hnilov}
\author{Alexander Fish}
\author{Leonid Yavits}
\end{frontmatter}

Section, table and figure numbers without the prefix S refer to the main text.

\section{Earlier single-seed round}\label{firstround}
The final accuracy-oriented settings were fixed after an earlier single-seed round, whose validation and test results were known at that time.
That round trained the FPTM models as reported in the main text, with at most 50 epochs.
On ALeRCE and NSL-KDD, it used 200 and 1,000 strict clauses per class, with $L=12$ for the 1,000-clause models, and 200 and 1,000 vanilla clauses per class with $s=10$. These baselines trained for at most 30 epochs.
On WOS-46985, it used 200 strict clauses per class for at most 30 epochs. Its vanilla TMs used 200 clauses per class with $s=25$ and 400 clauses per class with $s=10$, for at most 15 epochs. The 200-clause vanilla model was dropped.
After that round, we added the 400- and 600-clause strict models, unified $L=16$, gave all families the same epoch limit and kept $s=10$ for every vanilla TM.
The unification weakened the 1,000-clause strict baselines slightly. With $L=12$, the first-round ALeRCE model reached 93.33\% validation and 93.50\% test accuracy, against 93.13\% and 93.28\% for the reported run. On NSL-KDD, the corresponding test accuracies are 86.20\% and 86.02\% at equal validation accuracy.

\section{Training details and best epochs}\label{training}
""" + TRAIN + r"""
\textit{Best epochs.} Table~\ref{tab:hyperfull} extends Table~""" + LAB["tab:hyper"] + r""" of the main text with the best validation epoch of every run, listed by seed.

\begin{table}[H]
\footnotesize
\setlength{\tabcolsep}{4pt}
\centering
\caption{Accuracy-oriented hyperparameters, number of seeds and best epoch of each run. $S$: FPTM specificity divisor, which sets $d=\mathrm{round}(F/S)$. $s$: vanilla TM specificity. Strict: FPTM trainer with $\LF=1$.}
\label{tab:hyperfull}
""" + hyper + r"""\end{table}

\section{Additional modelling details}\label{modelling}
This section extends the assumptions of Section~""" + LAB["sec:eval"] + r""" of the main text.

\textit{Functional checks.} """ + EVAL + r"""
\textit{Refresh (M3).} \hyp{""" + REFR + r"""}

\textit{Retention tail (M3).} \hyp{""" + M3 + r"""}

\textit{Input loading (M4).} The load energy is excluded from the reported Mapping~S energy. """ + LOAD + r"""
\textit{Model store (M6).} \hyp{""" + M6 + r"""}

\textit{Model SRAM leakage (M7).} \hyp{""" + LEAK + r"""

\section{Accuracy versus energy of the accuracy-oriented models}\label{paretofull}
Fig.~\ref{fig:paretofull} plots the accuracy-oriented models of all three datasets. Fig.~""" + LAB["fig:front"] + r""" of the main text shows the ALeRCE and NSL-KDD models of both hyperparameter sets.

""" + FIG + r"""

\section{Mapping C results}\label{mapCres}
Table~\ref{tab:mapC} gives the full Mapping~C results of the accuracy-oriented models. Section~""" + LAB["sec:mapCres"] + r""" of the main text summarizes them.

""" + MAPC + r"""

\section{CPU inference}\label{cpu}
We ran every exported model on the CPU of our environment, an Intel Xeon processor at 2.10~GHz in a virtual machine with two virtual cores. Table~\ref{tab:cpu} lists the inference times. The virtual machine exposes no power counters, so CPU energy was not measured.

""" + CPUTAB.replace(r"\begin{table}[!htbp]", r"\begin{table}[H]") + r"""

""" + CPUBOUND + r"""

\textit{Kernels and timing protocol.} """ + CPUPROT + r"""

\section{Sensitivity analysis}\label{sens}
""" + sens + r"""
\section{Mapping S cycle count versus model size}\label{spearman}
For each dataset, Table~\ref{tab:spearman} relates the Mapping~S cycle count of the 11 FPTM configurations (five accuracy-oriented and six energy-oriented) to three measures of model size.
The measures are the clause count per class, the number of included literals per class, and this number weighted by $\log_2(\LF+1)$, which approximates the width of the miss counter.
The weighting is a descriptive choice made after the experiments.
The cycle count follows the weighted literal count closely (Spearman rank correlation """ + f"{lo_w:.2f} to {hi_w:.2f}" + r""") and the clause count only weakly (""" + f"{lo_c:.2f} to {hi_c:.2f}" + r""").

\begin{table}[H]
\centering
\caption{Spearman rank correlation between the Mapping~S cycle count and model size, per dataset, over the FPTM configurations of both hyperparameter sets.}
\label{tab:spearman}
\begin{tabular}{lrrrr}
\toprule
Dataset & Config. & Clauses & Literals & Lit.\ $\times\log_2(\LF+1)$ \\
\midrule
""" + sp + r"""
\bottomrule
\end{tabular}
\end{table}

\section{Additional discussion}\label{discussion}
""" + DISC + r"""
\end{document}
"""
open(os.path.join(OUT, "supplementary.tex"), "w").write(doc)
print("supplement written")
