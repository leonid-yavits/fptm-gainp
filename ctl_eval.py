"""
Adds the GainP controller to the hardware cost model (post-processing of
results/hw_eval_base.json, written by hw_eval.py, into results/hw_eval.json).

Controller model (ASSUMPTIONS, listed in the paper in blue):
- A sequencer expands the stored sparse model into the COMPARE/WRITE/READ
  stream on the fly. It issues one array operation per cycle.
- Sequencer energy: E_SEQ_PJ per issued cycle, including refresh cycles.
  Anchor: a 32-bit SRAM access costs 5 pJ in 45 nm CMOS (Han et al., ICLR 2016,
  arXiv 1510.00149). The 16 nm reference points use 5 pJ. The 7 nm corners apply
  only the 16->7 nm factors of the array (1/1.35 conservative, 0.35 optimistic),
  so the 45->16 nm reduction is not taken and the values remain upper estimates.
- Model store (Mapping S only): one 16-bit entry per included literal and one
  16-bit header per clause, in on-chip SRAM. Each 32-bit read returns two entries
  and costs E_RD_PJ (same anchor and scaling), independent of the store size.
  The model is read once per batch. The store is 1024 bits wide (32 reads per
  cycle); the average demand of every model stays below half of this width.
  A prefetch buffer large enough to avoid stalls is assumed (the longest strict
  clause has 288 literals, the longest vanilla clause 2,004); its area and
  energy are not modelled. Model-SRAM leakage is not modelled.
- Input loading (F write cycles per batch) is excluded, like its array energy;
  load_total_nJ bounds it (cell, sequencer and static energy).
- Macro: N_SUB subarrays share one sequencer and one model store. In Mapping S
  the macro holds floor(N_SUB / h) stacks of h subarrays, i.e. 64 samples per
  stack, and the per-batch controller energy is shared by all samples.
- SRAM area: entries x 16 bit x 7 nm high-density SRAM cell 0.027 um^2
  (Wu et al., IEDM 2016) / 0.75 array efficiency (efficiency of the GainP macro).
  Sequencer logic area is not modelled.
- Mapping C: the model is stored in the array. The sequencer issues the
  sample-dependent program, one sample at a time, so its energy is charged
  per sample without amortization.
"""
import json, sys

E_SEQ_PJ = 5.0
E_RD_PJ = 5.0
N_SUB = 256
SRAM_CELL_UM2 = 0.027
SRAM_EFF = 0.75
SUB_AREA_UM2 = 1889.9
CORNERS = {"sys": 1.0, "cir": 1.0, "n7c": 0.027 / 0.07, "n7o": 0.027 / 0.07, "n7r": 0.027 / 0.07}
ESCALE = {"sys": 1.0, "cir": 1.0, "n7c": 1 / 1.35, "n7o": 0.35, "n7r": 1 / 1.35}
STORE_WORDS_PER_CYCLE = 32
F_IN = {"alerce": 1172, "nslkdd": 178, "wos": 10000}


def ctl_terms(r):
    h = int(r["S_subarrays_per_col"])
    stacks = N_SUB // h
    entries = r["live_literals"] + r["clauses_total"]
    reads = (entries + 1) // 2
    sram_mm2 = entries * 16 * SRAM_CELL_UM2 / SRAM_EFF * 1e-6
    return stacks, reads, sram_mm2


def main(root="results"):
    import hw_eval as H
    R = json.load(open(f"{root}/hw_eval_base.json"))
    worst = 0.0
    for r in R:
        stacks, reads, sram = ctl_terms(r)
        h = int(r["S_subarrays_per_col"])
        F = F_IN[r["dataset"]]
        r["S_stacks_per_macro"] = stacks
        r["S_sram_mm2"] = sram
        r["S_model_reads"] = reads
        r["S_reads_per_cycle"] = reads / r["S_cycles"]
        worst = max(worst, r["S_reads_per_cycle"])
        for k, af in CORNERS.items():
            t_ns = H.MODELS[k]["t_cycle_ns"]
            es, er = E_SEQ_PJ * ESCALE[k], E_RD_PJ * ESCALE[k]
            s_cyc = r[f"S_{k}_batch_latency_us"] * 1e3 / t_ns          # includes refresh cycles
            c_cyc = r[f"C_{k}_latency_us"] * 1e3 / t_ns
            e_s = (es * s_cyc + er * reads) * 1e-3 / (64 * stacks)
            e_c = es * c_cyc * 1e-3
            r[f"S_{k}_E_ctl_nJ"] = e_s
            r[f"C_{k}_E_ctl_nJ"] = e_c
            r[f"S_{k}_energy_noctl_nJ"] = r[f"S_{k}_energy_nJ_per_sample"]
            r[f"S_{k}_energy_nJ_per_sample"] += e_s
            r[f"S_{k}_energy_percolumn_nJ"] += e_s
            r[f"C_{k}_energy_noctl_nJ"] = r[f"C_{k}_energy_nJ_per_sample"]
            r[f"C_{k}_energy_nJ_per_sample"] += e_c
            # input loading bound (excluded from the reported energy): cell + sequencer + static
            stat = H.MODELS[k]["static"] * 1e-6 * h * 256 / 1e6 * F * t_ns * 1e-9 * 1e9
            r[f"S_{k}_load_total_nJ"] = (r[f"S_{k}_load_energy_nJ_per_sample"]
                                       + es * F * 1e-3 / (64 * stacks) + stat)
            r[f"S_{k}_energy_withload_nJ"] = r[f"S_{k}_energy_nJ_per_sample"] + r[f"S_{k}_load_total_nJ"]
            arr = stacks * h * SUB_AREA_UM2 * af * 1e-6
            sram_k = sram if k.startswith("n7") else sram * 0.07 / 0.027
            r[f"S_{k}_macro_area_mm2"] = arr + sram_k
            t = r[f"S_{k}_batch_latency_us"] + r[f"S_{k}_load_latency_us"]
            r[f"S_{k}_macro_throughput"] = stacks * 64 / (t * 1e-6)
            r[f"S_{k}_throughput_per_mm2"] = r[f"S_{k}_macro_throughput"] / (arr + sram_k)
        r["S_E_ctl_nJ"] = r["S_n7c_E_ctl_nJ"]
        r["C_E_ctl_nJ"] = r["C_n7c_E_ctl_nJ"]
    assert worst <= STORE_WORDS_PER_CYCLE, worst
    json.dump(R, open(f"{root}/hw_eval.json", "w"), indent=1)
    print("controller terms added to", len(R), "runs; max reads per cycle %.2f" % worst)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
