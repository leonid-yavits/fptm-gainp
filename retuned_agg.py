"""Aggregate the energy-oriented (retuned) runs and compute accuracy/energy fronts.
Runs: ALeRCE and WOS-46985 = run A (300 epochs) + run B (min 300 epochs, patience 50);
NSL-KDD = one run. Model selection: best validation epoch (as in the main experiments).
Writes results/retuned_agg.json."""
import json
from collections import defaultdict
fam = lambda r: 'Vanilla' if r['tag'].startswith('V') else ('Strict' if r['LF'] == 1 else 'FPTM')
RUNS = {'alerce': ['results_alerce_ah', 'results_alerce_long'],
        'wos': ['results_wos_ah', 'results_wos_long'],
        'nslkdd': ['results_nslkdd_ah']}
KEYS = ['test_acc', 'S_n7c_energy_nJ_per_sample', 'S_n7o_energy_nJ_per_sample', 'S_cycles',
        'C_subarrays', 'C_n7c_energy_nJ_per_sample', 'C_n7o_energy_nJ_per_sample', 'C_n7c_latency_us']
def agg(rows):
    out = {}
    for (f, c), L in rows.items():
        r0 = L[0]
        e = dict(family=f, clauses=c, n=len(L), T=r0['T'], S=r0['S'], L=r0['L'], LF=r0['LF'],
                 include_limit=r0['include_limit'],
                 lit_per_clause=sum(x['live_literals'] for x in L) / len(L) / r0['clauses_total'])
        for k in KEYS:
            v = [x[k] for x in L]; e[k] = sum(v) / len(v); e[k + '_min'] = min(v); e[k + '_max'] = max(v)
        out[f'{f}-{c}'] = e
    return out
def front(models, ek):
    # models on the accuracy/energy front: no other model has >= accuracy and <= energy (one strictly)
    F = []
    for a, m in models.items():
        if not any((o['test_acc'] >= m['test_acc'] and o[ek] <= m[ek]) and (o['test_acc'] > m['test_acc'] or o[ek] < m[ek])
                   for b, o in models.items() if b != a):
            F.append(a)
    return sorted(F, key=lambda a: models[a][ek])
res = {}
main = json.load(open('results/hw_eval.json'))
for ds, roots in RUNS.items():
    g = defaultdict(list)
    for r in main:
        if r['dataset'] == ds: g[(fam(r), r['clauses_per_class'])].append(r)
    A = agg(g)
    g = defaultdict(list)
    for root in roots:
        for r in json.load(open(f'{root}/hw_eval.json')): g[(fam(r), r['clauses_per_class'])].append(r)
    E = agg(g)
    allm = {**{'A:' + k: v for k, v in A.items()}, **{'E:' + k: v for k, v in E.items()}}
    res[ds] = dict(accuracy_oriented=A, energy_oriented=E,
                   front_opt=front(allm, 'S_n7o_energy_nJ_per_sample'),
                   front_cons=front(allm, 'S_n7c_energy_nJ_per_sample'))
json.dump(res, open('results/retuned_agg.json', 'w'), indent=1)
for ds in res:
    print(ds, 'front opt:', res[ds]['front_opt']); print(ds, 'front cons:', res[ds]['front_cons'])
    for k, e in sorted(res[ds]['energy_oriented'].items(), key=lambda kv: (kv[1]['family'] != 'FPTM', kv[1]['clauses'])):
        print('  E', k, 'n=%d acc %.2f [%.2f,%.2f] nJ %.2f/%.3f lit %.1f Csub %d IL %d' % (e['n'], 100*e['test_acc'], 100*e['test_acc_min'], 100*e['test_acc_max'], e['S_n7c_energy_nJ_per_sample'], e['S_n7o_energy_nJ_per_sample'], e['lit_per_clause'], e['C_subarrays'], e['include_limit']))
