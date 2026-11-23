"""Recomputes every number quoted in the paper prose (abstract, results,
discussion, conclusions) from results/hw_eval_agg.json and the model files."""
import json, pickle
A = {(a["dataset"], a["tag"]): a for a in json.load(open("results/hw_eval_agg.json"))}
R = json.load(open("results/hw_eval.json"))
g = lambda ds, t: A[(ds, t)]
pct = lambda ds, t: f"{100*g(ds,t)['test_acc']:.2f}" + (f" ± {100*g(ds,t)['test_acc_std']:.2f}" if g(ds,t).get('test_acc_std') else "")
E = lambda ds, t, c: round(g(ds, t)[f"S_{c}_energy_nJ_per_sample"], 3)
cyc = lambda ds, t: round(g(ds, t)["S_cycles"] / 1e3, 1)
print("runs:", len(R), "configs:", len(A))
print("ALeRCE test F2/F20/S600/S1000/V200/V1000:", pct("alerce","F1_C2"), pct("alerce","F3_C20"), pct("alerce","S_LF1_C600"), pct("alerce","S_LF1_C1000"), pct("alerce","V_C200"), pct("alerce","V_C1000"))
print("ALeRCE lits F20, S600, ratio:", round(g("alerce","F3_C20")["literals"]), round(g("alerce","S_LF1_C600")["literals"]), round(g("alerce","S_LF1_C600")["literals"]/g("alerce","F3_C20")["literals"],1))
for t in ("F1_C2","F2_C10","F3_C20","F4_C40","S_LF1_C200","S_LF1_C400","S_LF1_C600","S_LF1_C1000"):
    print("ALeRCE", t, pct("alerce",t), "cyc k", cyc("alerce",t), "E n7c/n7o", E("alerce",t,"n7c"), E("alerce",t,"n7o"))
f, s = g("alerce","F3_C20"), g("alerce","S_LF1_C600")
print("ALeRCE F20 vs S600: cycles ratio", round(s["S_cycles"]/f["S_cycles"],2), "energy ratio n7c", round(s["S_n7c_energy_nJ_per_sample"]/f["S_n7c_energy_nJ_per_sample"],2), "n7o", round(s["S_n7o_energy_nJ_per_sample"]/f["S_n7o_energy_nJ_per_sample"],2))
f2, s2 = g("alerce","F1_C2"), g("alerce","S_LF1_C200")
print("ALeRCE F2 vs S200: acc diff", round(100*(s2["test_acc"]-f2["test_acc"]),2), "cyc ratio", round(s2["S_cycles"]/f2["S_cycles"],2), "E ratio n7c", round(s2["S_n7c_energy_nJ_per_sample"]/f2["S_n7c_energy_nJ_per_sample"],2), "n7o", round(s2["S_n7o_energy_nJ_per_sample"]/f2["S_n7o_energy_nJ_per_sample"],2))
nsl = [(t, 100*a["test_acc"], 100*a["val_acc"]) for (d,t),a in A.items() if d=="nslkdd" and not t.startswith("V") and t!="F1_C2"]
print("NSL test range:", round(min(x[1] for x in nsl),2), round(max(x[1] for x in nsl),2), "val range:", round(min(x[2] for x in nsl),2), round(max(x[2] for x in nsl),2))
print("NSL V200/V1000:", pct("nslkdd","V_C200"), pct("nslkdd","V_C1000"), "S1000:", pct("nslkdd","S_LF1_C1000"))
for t in ("S_LF1_C200","F3_C20","F4_C40","S_LF1_C600"):
    print("NSL", t, pct("nslkdd",t), "cyc k", cyc("nslkdd",t), "E", E("nslkdd",t,"n7c"), E("nslkdd",t,"n7o"))
for t in ("F1_C2","F5_C100","S_LF1_C200","S_LF1_C400"):
    print("WOS", t, pct("wos",t), "cyc k", cyc("wos",t), "E", E("wos",t,"n7c"), E("wos",t,"n7o"))
for k,a in A.items():
    if k[0]=="wos" and k[1].startswith("V"): print("WOS vanilla", k[1], pct("wos",k[1]))
lits = [r for r in R if r["dataset"]=="wos" and r["tag"]=="F1_C2"]
print("WOS F2 literals per clause:", [round(r["literals"]/r["clauses_total"]) for r in lits])
m = pickle.load(open("results/wos/F1_C2_s0/model.pkl","rb"))["model"]
print("WOS F2 s0 plain/negated:", sum(len(c["lits"]) for c in m), sum(len(c["lits_neg"]) for c in m))
n1 = [r for r in R if r["dataset"]=="nslkdd" and r["tag"]=="F1_C2" and r["seed"]==0][0]
print("NSL F2 s0 conflicts:", n1["conflict_clauses"], "/", n1["clauses_total"])
F, S = g("alerce","F3_C20"), g("alerce","S_LF1_C1000")
print("MapC ALeRCE S1000 cols/area n7o:", S["C_columns"], round(S["C_n7o_area_mm2"],3), "F20 area/cycles/cols:", round(F["C_n7o_area_mm2"],4), round(F["C_cycles"]), F["C_columns"], "area ratio", round(S["C_n7o_area_mm2"]/F["C_n7o_area_mm2"],1), "red ratio", round(S["C_columns"]/F["C_columns"],1))
worse = [(k, round(a["C_n7c_energy_nJ_per_sample"]/a["S_n7c_energy_nJ_per_sample"],2)) for k,a in A.items() if a["C_n7c_energy_nJ_per_sample"] <= a["S_n7c_energy_nJ_per_sample"]]
print("configs where Mapping C energy <= Mapping S (n7c):", worse)
worse_o = [k for k,a in A.items() if a["C_n7o_energy_nJ_per_sample"] <= a["S_n7o_energy_nJ_per_sample"]]
print("configs where Mapping C energy <= Mapping S (n7o):", worse_o)
eqc = [(t, a["test_acc"], A.get(("wos" if False else d, "V"+t[5:]) , {}).get("test_acc")) for (d,t),a in A.items() if t.startswith("S_LF1")]
for (d,t),a in A.items():
    if t.startswith("S_LF1"):
        v = A.get((d, "V_C" + t.split("_C")[1]))
        if v: print("strict vs vanilla", d, t, round(100*a["test_acc"],2), round(100*v["test_acc"],2))
