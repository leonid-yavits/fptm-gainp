"""
GainP hardware evaluation of all trained models (FPTM and strict TM baselines),
one row per training run (dataset, configuration, seed).

GainP constants come from the GainP article (Levi et al., J. Syst. Archit. 172,
103632, 2026; local final version GainP-final-clean-version.tex) or from its
evaluation workbook ("GC-AP SpMM Performance and Energy (v9 SOTA scaled to
16nm).xlsx"). Everything else is marked ASSUMPTION and is listed in the paper.
"""
import csv, glob, json, os, pickle, sys
import numpy as np
from datasets import load
from fptm import reference_votes, strict_votes
from gainp_sim import run_sample_stationary, bitlen, normalize_clause
from gainp_clause import run_clause_stationary

# ---- GainP parameters --------------------------------------------------------
CAM_ROWS_PER_SUBARRAY = 128       # 256 x 64 gain cells = 128 x 64 CAM cells (article)
COLS_PER_SUBARRAY = 64
GAIN_CELLS_PER_SUBARRAY = 256 * 64
SUBARRAY_AREA_UM2 = 1889.9        # one 256x64 macro (article)
STATIC_16_UW_PER_MBIT = 380.0     # evaluation workbook, sheet "GAP Power Calc"
# 16 nm -> 7 nm scaling from published foundry data (Wu et al., IEDM 2016,
# 7 nm vs the same foundry's 16 nm FinFET): 35-40 % speed gain at equal power,
# or > 65 % power reduction at equal speed; 6T HD SRAM cell 0.027 um^2 at 7 nm
# vs about 0.07 um^2 at 16 nm (Wu et al., IEDM 2013).
SPEED_GAIN_ISO_POWER = 1.35       # conservative end of 35-40 %
POWER_FACTOR_ISO_SPEED = 0.35     # > 65 % reduction
AREA_FACTOR = 0.027 / 0.07        # HD SRAM cell area ratio, applied to the whole macro
MODELS = {
    # 16 nm reference points (GainP article and evaluation workbook)
    "sys": dict(t_cycle_ns=1.0, e_cell_fJ=2.17, area=1.0, static=STATIC_16_UW_PER_MBIT),
    "cir": dict(t_cycle_ns=2.0, e_cell_fJ=10.4, area=1.0, static=STATIC_16_UW_PER_MBIT),
    # PREDICTED 7 nm, conservative: circuit point, iso-power speed gain
    # (cycle / 1.35, energy per operation / 1.35), static power per Mbit unchanged
    "n7c": dict(t_cycle_ns=2.0 / SPEED_GAIN_ISO_POWER, e_cell_fJ=10.4 / SPEED_GAIN_ISO_POWER,
                area=AREA_FACTOR, static=STATIC_16_UW_PER_MBIT),
    # PREDICTED 7 nm, optimistic: system point, iso-speed power reduction
    # (cycle unchanged, energy and static power x 0.35)
    "n7o": dict(t_cycle_ns=1.0, e_cell_fJ=2.17 * POWER_FACTOR_ISO_SPEED,
                area=AREA_FACTOR, static=STATIC_16_UW_PER_MBIT * POWER_FACTOR_ISO_SPEED),
    # PREDICTED 7 nm sensitivity: conservative point with 10x shorter retention
    "n7r": dict(t_cycle_ns=2.0 / SPEED_GAIN_ISO_POWER, e_cell_fJ=10.4 / SPEED_GAIN_ISO_POWER,
                area=AREA_FACTOR, static=STATIC_16_UW_PER_MBIT, ret=0.1),
}
# ASSUMPTION: refresh every mean - 3 sigma of the simulated retention time
# (75.67 us, sigma 7.57 us). One refresh reads and rewrites all 256 gain-cell
# rows of every occupied subarray in lockstep (512 cycles), 2 cell operations
# per CAM cell. Applied identically to both mappings, amortized over time.
T_REF_US = 75.67 - 3 * 7.57
REFRESH_CYCLES = 256 * 2
# ASSUMPTIONS: a WRITE and a READ take one cycle and cost the same per-cell
# energy as a COMPARE; dynamic energy scales with the number of unmasked rows
# (masked rows cost nothing); Mapping S input loading writes one input row per
# cycle (F cycles per batch), off-chip I/O is not modelled.

_DATA = {}


def _data(ds):
    if ds not in _DATA:
        _DATA[ds] = load(ds)
    return _DATA[ds]


def _refresh(lat_s, t, cells_cam, ret=1.0):
    """Refresh amortized over time: in every refresh period the array stalls
    for REFRESH_CYCLES and performs 2 operations per occupied CAM cell. The
    share charged to an operation of duration lat_s is lat_s / T_ref."""
    n_ref = lat_s / (T_REF_US * ret * 1e-6)
    return n_ref, n_ref * REFRESH_CYCLES * t, n_ref * 2 * cells_cam


def eval_model(path):
    run_dir = os.path.dirname(path)
    M = pickle.load(open(path, "rb"))
    S = json.load(open(os.path.join(run_dir, "summary.json")))
    strict = bool(M.get("strict", False))
    LF = M["LF"] or 1
    F = M["n_features"]
    d = _data(S["dataset"])
    # accuracy of the exported model, i.e. the model that GainP executes
    V = (strict_votes(M["model"], d["Xte"], M["n_classes"]) if strict
         else reference_votes(M["model"], d["Xte"], M["n_classes"], LF))
    acc_mapped = float((np.argmax(V, 1) == d["yte"]).mean())

    p, _ = run_sample_stationary(M["model"], M["n_classes"], LF, strict, F, X=None)
    R = p.n_rows
    h = -(-R // CAM_ROWS_PER_SUBARRAY)
    pc, _, ci = run_clause_stationary(M["model"], M["n_classes"], LF, strict, X=None)
    hC = -(-pc.n_rows // CAM_ROWS_PER_SUBARRAY)
    wC = -(-max(ci["columns"], 1) // COLS_PER_SUBARRAY)
    nsubC = hC * wC

    lits = int(S["total"])
    nrm = [normalize_clause(m, LF, strict) for m in M["model"]]
    out = dict(dataset=S["dataset"], tag=S["tag"], seed=int(S.get("seed", 0)),
               kind="strict" if (strict or LF == 1) else "FPTM",
               impl=S.get("impl", "FPTM port"), clauses_per_class=S["clauses"],
               LF=None if strict else LF, L=S.get("L"), T=S["T"], S=S.get("S"), s=S.get("s"),
               include_limit=S.get("include_limit"), best_epoch=S["best_epoch"],
               epochs_run=S.get("epochs_run"), max_epochs=S["epochs"],
               val_acc=S["val_acc"], test_acc=acc_mapped, test_acc_trainer=S["test_acc"],
               literals=lits, live_literals=sum(len(r) for r, b, k in nrm if r is not None),
               clauses_total=len(M["model"]),
               conflict_clauses=sum(1 for m in M["model"] if len(set(m["lits"]) & set(m["lits_neg"])) > 0),
               removed_clauses=sum(1 for r, b, k in nrm if r is None),
               unit_clauses=sum(1 for r, b, k in nrm if r is not None and b == 1),
               fuzzy_clauses=sum(1 for r, b, k in nrm if r is not None and b > 1),
               model_bytes_sparse=int(np.ceil(lits * (bitlen(F - 1) + 1) / 8)),
               S_cycles=p.cycles, S_compares=p.n_cmp, S_writes=p.n_wr, S_reads=p.n_rd,
               S_rows=R, S_subarrays_per_col=h, S_act=p.act,
               C_cycles=pc.cycles, C_compares=pc.n_cmp, C_writes=pc.n_wr, C_reads=pc.n_rd,
               C_columns=ci["columns"], C_F_used=ci["F_used"], C_rows=pc.n_rows,
               C_subarrays=nsubC, C_act=pc.act,
               C_reduction_adds=ci["reduction_adds"], C_strict_program=ci["strict_program"])
    for k, m in MODELS.items():
        t = m["t_cycle_ns"] * 1e-9; e = m["e_cell_fJ"] * 1e-15
        STATIC_UW_PER_MBIT = m["static"]
        area_per64_mm2 = h * SUBARRAY_AREA_UM2 * m["area"] * 1e-6
        areaC_mm2 = nsubC * SUBARRAY_AREA_UM2 * m["area"] * 1e-6
        out[f"C_{k}_area_mm2"] = areaC_mm2
        # Mapping S: one column per sample, h stacked subarrays
        lat = p.cycles * t
        n_ref, t_ref, ops_ref = _refresh(lat, t, h * CAM_ROWS_PER_SUBARRAY, m.get("ret", 1.0))
        lat_tot = lat + t_ref
        e_dyn = p.act * e
        e_ref = ops_ref * e
        e_stat = STATIC_UW_PER_MBIT * 1e-6 * (h * 256 / 1e6) * lat_tot
        out.update({
            f"S_{k}_batch_latency_us": lat_tot * 1e6,
            f"S_{k}_refresh_overhead_pct": 100 * t_ref / lat,
            f"S_{k}_E_dyn_nJ": e_dyn * 1e9,
            f"S_{k}_E_ref_nJ": e_ref * 1e9,
            f"S_{k}_E_static_nJ": e_stat * 1e9,
            f"S_{k}_energy_nJ_per_sample": (e_dyn + e_ref + e_stat) * 1e9,
            f"S_{k}_energy_percolumn_nJ": (p.cycles * e + e_ref + e_stat) * 1e9,   # sensitivity: rows ignored
            f"S_{k}_load_energy_nJ_per_sample": F * e * 1e9,
            f"S_{k}_load_latency_us": F * t * 1e6,
            f"S_{k}_throughput_per_mm2": 64 / area_per64_mm2 / (lat_tot + F * t),
        })
        # Mapping C: whole model array, one sample at a time
        latC = pc.cycles * t
        n_refC, t_refC, ops_refC = _refresh(latC, t, nsubC * CAM_ROWS_PER_SUBARRAY * COLS_PER_SUBARRAY, m.get("ret", 1.0))
        latC_tot = latC + t_refC
        eC_dyn = pc.act * ci["columns"] * e
        eC_ref = ops_refC * e
        eC_stat = STATIC_UW_PER_MBIT * 1e-6 * (nsubC * GAIN_CELLS_PER_SUBARRAY / 1e6) * latC_tot
        out.update({
            f"C_{k}_latency_us": latC_tot * 1e6,
            f"C_{k}_E_dyn_nJ": eC_dyn * 1e9,
            f"C_{k}_E_ref_nJ": eC_ref * 1e9,
            f"C_{k}_E_static_nJ": eC_stat * 1e9,
            f"C_{k}_energy_nJ_per_sample": (eC_dyn + eC_ref + eC_stat) * 1e9,
            f"C_{k}_throughput_per_mm2": 1.0 / latC_tot / areaC_mm2,
        })
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    rows = []
    for path in sorted(glob.glob(f"{root}/*/*/model.pkl")):
        r = eval_model(path)
        rows.append(r)
        print(r["dataset"], r["tag"], r["seed"], r["kind"], "test", round(r["test_acc"], 4),
              "S cyc", r["S_cycles"], "S nJ", round(r["S_sys_energy_nJ_per_sample"], 3),
              "C cyc", r["C_cycles"], flush=True)
    with open(f"{root}/hw_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump(rows, open(f"{root}/hw_eval.json", "w"), indent=1)
