"""LaTeX rows for the energy-oriented runs (Table 10: ALeRCE and NSL-KDD; Table 11: WOS-46985)."""
import json
R = json.load(open("results/retuned_agg.json"))
def fmt_sub(v):
    return f"{v:,.0f}" if v == int(v) else f"{v:,.1f}"
def rows(ds):
    E = R[ds]["energy_oriented"]; out = []
    for k, e in sorted(E.items(), key=lambda kv: (kv[1]["family"] != "FPTM", kv[1]["clauses"])):
        name = f"{e['family']} ({e['clauses']:,})"
        out.append(f"{name} & {e['T']} & {e['S']:,} & {e['L']} & {e['LF']} & {100*e['test_acc']:.2f} & {e['lit_per_clause']:.1f} & "
                   f"{e['S_cycles']/1e3:,.1f} & {e['S_n7c_energy_nJ_per_sample']:.2f} & {e['S_n7o_energy_nJ_per_sample']:.3f} & {fmt_sub(e['C_subarrays'])} \\\\")
    return out
open("paper/tables/energyhp.tex", "w").write("\n".join([r"\multicolumn{11}{l}{\emph{ALeRCE (two runs)}}\\"] + rows("alerce") +
    [r"\midrule", r"\multicolumn{11}{l}{\emph{NSL-KDD (one run)}}\\"] + rows("nslkdd") +
    [r"\midrule", r"\multicolumn{11}{l}{\emph{WOS-46985 (two runs)}}\\"] + rows("wos")) + "\n")
open("paper/tables/woshp.tex", "w").write("\n".join(rows("wos")) + "\n")
for ds in ("alerce", "wos"):
    E = R[ds]["energy_oriented"]
    da = max(100*(e["test_acc_max"]-e["test_acc_min"]) for e in E.values())
    de = max((e["S_n7o_energy_nJ_per_sample_max"]-e["S_n7o_energy_nJ_per_sample_min"])/e["S_n7o_energy_nJ_per_sample"] for e in E.values())
    dc = max((e["S_n7c_energy_nJ_per_sample_max"]-e["S_n7c_energy_nJ_per_sample_min"])/e["S_n7c_energy_nJ_per_sample"] for e in E.values())
    print(ds, "max acc spread %.2f pts, max energy spread opt %.1f%% cons %.1f%%" % (da, 100*de, 100*dc))
print(open("paper/tables/energyhp.tex").read()); print(open("paper/tables/woshp.tex").read())
