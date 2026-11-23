# FPTM on GainP

Code, results, manuscript sources and submission files of the paper

> **Fuzzy-Pattern Tsetlin Machine Inference on a Gain Cell Embedded DRAM Associative In-Memory Processor**
> Artem Hnilov, Alexander Fish, Leonid Yavits.
> Submitted to *Expert Systems with Applications* (Elsevier), manuscript ESWA-S-26-70331, November 2026.

This README is the handout for anyone who takes over the work. It explains what the paper claims, how every number is produced, where each file lives, and how to rebuild the submission.

## 1. The work in one page

**Question.** The Fuzzy-Pattern Tsetlin Machine (FPTM) lets a clause vote with a reduced score when some of its literals fail. With accuracy-oriented hyperparameters it reaches the accuracy of the best strict Tsetlin Machine (TM) with 15x to 25x fewer clauses per class (ALeRCE, NSL-KDD). Does this advantage survive in-memory inference on GainP, a gain cell embedded DRAM (GC-eDRAM) associative processor?

**What was done.**
1. FPTM inference is mapped onto the GainP COMPARE, WRITE and READ primitives in two layouts:
   * **Mapping S** (sample-stationary): 64 input samples per logical column group, clauses broadcast as compare programs.
   * **Mapping C** (clause-stationary): clauses stored as ternary patterns in the array.
2. An exact compile-time clause simplification removes redundant literals. Two in-array miss counters (thermometer and binary) implement the fuzzy vote.
3. A functional simulator executes the generated programs on real test samples. The in-array predictions match an independent NumPy reference on all checked samples (1,500 per run for Mapping S, 200 for Mapping C).
4. FPTM, strict (FPTM trainer with LF = 1) and vanilla TM models are trained on WOS-46985 (text), ALeRCE (astronomical light curves) and NSL-KDD (intrusion records).
5. Latency, energy and area are evaluated for a new 7 nm FinFET GainP design (not fabricated) at a conservative and an optimistic corner, with the 16 nm GainP values as a reference.

**Headline results.**
* Accuracy-oriented hyperparameters: Mapping S energy per classification is 0.15 to 112 nJ including the controller. This lies within the range of published TM ASICs and in-memory designs and below TM FPGA accelerators. Under a 1 W core-power bound, a single-thread CPU needs at least 63x more energy.
* A fuzzy clause costs a bit-serial miss count, so FPTM energy follows its number of included literals, not its clause count.
* Energy-oriented hyperparameters cut FPTM energy by 1.6x to 25x. On NSL-KDD, FPTM then reaches the accuracy band of the best strict model at 13x to 23x less energy. On ALeRCE, FPTM gives the cheapest points. On WOS-46985, strict models stay cheaper.

## 2. Repository layout

| Path | Content |
|---|---|
| `fptm.py` | Python/Numba line-by-line port of the multiclass FPTM reference (`src/FuzzyPatternTM.jl` in BooBSD/FuzzyPatternTM). Also `reference_votes` (NumPy FPTM inference) and `strict_votes` (strict TM inference). |
| `prepare_data.py` | WOS-46985 booleanization: binary unigrams and bigrams, chi2 top 10,000, fitted on the training split (including validation) only. |
| `datasets.py` | Loads the three datasets as Boolean train, validation and test splits. |
| `run_fptm.py` | Trains FPTM (or strict with LF = 1), selects the epoch on validation, exports the clause literal lists (`model.pkl`). Environment variables: `PATIENCE` (0 disables early stopping), `MIN_EPOCHS`, `CKPT_EVERY`, `RESULTS_ROOT`. |
| `run_vanilla.py` | Vanilla TM baseline through pyTsetlinMachine, same protocol, export checked against `predict`. |
| `gainp_sim.py` | Mapping S compiler (clause simplification, miss counters, vote accumulation, signed argmax) and functional simulator. |
| `gainp_clause.py` | Mapping C program generator and functional simulator with ternary storage. |
| `check_gainp.py`, `check_all.sh` | Bit-exact checks of both mappings against the software reference. |
| `hw_eval.py` | Cycles, latency, energy, area and throughput of every run at the 16 nm reference points and the 7 nm corners. Writes `<root>/hw_eval.json`. |
| `ctl_eval.py` | Adds the sequencer and model SRAM (controller) terms. Reads `<root>/hw_eval_base.json`, writes `<root>/hw_eval.json`. |
| `cpu_bench.py` | CPU baseline: three single-thread Numba kernels per model (dense bit-packed, sparse with early exit, 64-sample bit-sliced), each verified against the reference; the fastest is reported. |
| `make_tables.py` | Writes `paper/tables/*.tex` from `results/hw_eval.json` (mean and sample std over seeds). No table number is typed by hand. |
| `retuned_agg.py` | Aggregates the energy-oriented runs and computes the accuracy/energy fronts. Writes `results/retuned_agg.json`. |
| `make_retuned_tables.py` | Rows of the energy-oriented tables (`paper/tables/energyhp.tex`, `woshp.tex`). |
| `make_front_fig.py` | Fig. 3, `paper/figures/acc_energy_front.pdf`. |
| `make_figs.py` | Supplementary figure `acc_vs_energy_S.pdf`. |
| `audit_numbers.py` | Recomputes every number quoted in the prose (abstract, results, discussion, conclusions). |
| `sweep*.txt` | One command line per training run (see Section 4). |
| `run_queue.sh`, `run_queue_long.sh`, `poll_*.sh` | Two-jobs-at-a-time queue runners with resume from `ckpt.pkl`. |
| `final_epoch_eval.py` | Scores the epoch-300 state instead of the best validation epoch. Not used for the paper. |
| `test_*.py` | Early sanity tests of the port on scikit-learn digits. |
| `assemble.py`, `build_supp.py`, `build_check.py`, `build_all.sh`, `paper/sections/` | **Legacy** pipeline that assembled the first single-file Overleaf project from section files. The manuscript has since been edited directly on Overleaf. `manuscript/` is authoritative. |
| `results/` | Main experiments (accuracy-oriented hyperparameters, 80 runs, 3 seeds where listed). |
| `results_{wos,alerce}_ah/`, `results_{wos,alerce}_long/`, `results_nslkdd_ah/` | Energy-oriented runs (Section 5.7). |
| `data/`, `inputs/treetm/` | Prepared Boolean datasets (Section 3). |
| `paper/tables/`, `paper/figures/` | Generated tables and figures. |
| `manuscript/` | Final Overleaf sources: `main.tex`, `supplementary.tex`, `refs.bib`, `figures/`. |
| `submission/` | The ESWA submission files and the scripts that rebuild them (Section 7). |
| `reviews/` | Logs of the last three independent internal reviews (19 to 21). |

Each run directory `<root>/<dataset>/<tag>_s<seed>/` holds `summary.json` (configuration, best epoch, validation and test accuracy, model size), `log.jsonl` (per epoch), `model.pkl` (exported clauses) and `bitexact.txt` or `check.txt` (bit-exact check output).

## 3. Data

| Dataset | Classes | Boolean inputs | Train / val / test | Source of the splits |
|---|---|---|---|---|
| WOS-46985, level 1 | 7 | 10,000 | 80/20 stratified train/test, 10% of train held out for validation | `prepare_data.py` → `data/wos_WOS46985_YL1_k10000.npz` |
| ALeRCE | 15 leaf classes | 1,172 | 88,916 / 9,880 / 24,700 | TreeTM preparation code, seed 0 → `inputs/treetm/TreeTM_datasets/alerce/` |
| NSL-KDD | 23 | 178 | 113,375 / 12,598 / 18,794 (KDDTest+ restricted to attack types seen in training) | TreeTM preparation code, seed 0 → `inputs/treetm/TreeTM_datasets/nslkdd/` |

The prepared arrays are committed, so every step after training runs without the raw data. To rebuild WOS from scratch, download `WebOfScience.zip` (Kowsari et al., Mendeley Data), extract it, and run

```
python3 prepare_data.py wos <WebOfScience folder> --subset WOS46985 --level YL1 --k 10000
```

The validation split of WOS is drawn inside `datasets.py` from the training split. Model selection (epoch, hyperparameters) uses validation only. Test accuracy is reported once.

## 4. Training runs

Tags: `F<k>_C<c>` = FPTM configuration k with c clauses per class; `S_LF1_C<c>` = strict (LF = 1); `V_C<c>` = vanilla TM; prefix `AH_` = energy-oriented hyperparameters. Command format:

```
python3 run_fptm.py   <dataset> <tag> clauses T S L LF include_limit max_epochs [seed]
python3 run_vanilla.py <dataset> <tag> clauses_per_class T s max_epochs [seed]
```

| Sweep file | Runs | Paper use |
|---|---|---|
| `sweep2.txt` | 80 | Main experiments (Tables 2, 4, 5, 6; max 50 epochs, patience 10) → `results/` |
| `sweep_wos_ah.txt`, `sweep_alerce_ah.txt`, `sweep_nslkdd_ah.txt` | 8, 10, 10 | Energy-oriented run A (300 epochs, no early stopping) → `results_*_ah/` |
| `sweep_long.txt` | 18 | Energy-oriented run B for WOS and ALeRCE (min 300 epochs, patience 50) → `results_*_long/` |
| `sweep.txt`, `sweep3.txt`, `sweep_rest.txt`, `sweep_s1.txt` | | Earlier single-seed round (Supplementary Section S1). Results not committed. |
| `sweep_extra.txt` | 7 | Include-limit 200 probe on ALeRCE. Not used. |

Run a sweep two jobs at a time:

```
sh -c 'cat sweep2.txt | xargs -P 2 -I{} sh -c "{}"'
```

## 5. Hardware model

### Parameters (Table 3)

| Parameter | 16 nm circuit | 16 nm system | 7 nm conservative | 7 nm optimistic |
|---|---|---|---|---|
| Compare cycle (ns) | 2.00 | 1.00 | 1.48 | 1.00 |
| Energy per cell operation (fJ) | 10.40 | 2.17 | 7.70 | 0.76 |
| Subarray macro area (um^2) | 1890 | 1890 | 729 | 729 |
| Gain-cell area (um^2) | 0.0820 | 0.0820 | 0.0316 | 0.0316 |
| Static power (uW/Mbit) | 380 | 380 | 380 | 133 |
| Refresh period (us) | 52.96 | 52.96 | 52.96 | 52.96 |

The 16 nm values come from the GainP article (Levi, Harel, Teman, Yavits, J. Syst. Archit. 172, 103632, 2026) and its evaluation workbook. The 7 nm values are those of the 7 nm GainP design. In `hw_eval.py` (dictionary `MODELS`, keys `sys`, `cir`, `n7c`, `n7o`, `n7r`) the 7 nm corners are written as factors on the 16 nm points; the resulting values are those of the table. `n7r` is the sensitivity point with 10x shorter retention.

Array organization: 256x64 gain-cell subarrays (128x64 CAM cells). A logical column stacks h subarrays whose read bitlines are joined during COMPARE, with local sense amplifiers and tags kept for refresh.

### Modelling rules (M1 to M7 in Section 4.3)

* Latency = cycles x cycle time, plus refresh stalls.
* Dynamic energy = sum over cycles of unmasked rows x active columns x energy per cell operation. Masked rows cost nothing (M2). The alternative of one cell operation per column and cycle is in the sensitivity analysis.
* WRITE and READ take one cycle and cost the same per-cell energy as COMPARE.
* Refresh every 52.96 us (mean minus 3 sigma of the simulated retention, 75.67 us and 7.57 us). One refresh rewrites all rows of every occupied subarray in lockstep, 512 cycles, two cell operations per CAM cell, amortized (M3).
* Mapping S loads a batch with one input row per cycle, F cycles. Off-chip input bandwidth is not modelled (M4).
* Controller (M6, `ctl_eval.py`): a sequencer generates the COMPARE/WRITE/READ stream. Energy per issued cycle and per 32-bit model store read, anchored at 5 pJ (45 nm 32-bit SRAM access, Han et al. 2016); 3.70 pJ and 1.75 pJ at the conservative and optimistic 7 nm corners. One sequencer and model store serve a macro of 256 subarrays in Mapping S, shared by all samples of the macro. In Mapping C the controller energy is charged per sample.
* Mapping C excludes the cross-column vote reduction.

The cycle and row counts of each program do not depend on the data.

## 6. Reproducing the paper

Requirements: Python 3.10 or later, `numba scikit-learn pandas pyarrow matplotlib pyTsetlinMachine python-docx`, and TeX Live with `latexmk` for the manuscript. Tested with Python 3.13, Numba 0.68 and scikit-learn 1.9.

### From the committed models (minutes)

```
# bit-exact checks of one run (Mapping S on 1,500, Mapping C on 200 test samples)
python3 check_gainp.py nslkdd F1_C2_s0 1500 200
# all main runs without a check result
./check_all.sh
# hardware evaluation of the main runs
python3 hw_eval.py && cp results/hw_eval.json results/hw_eval_base.json && python3 ctl_eval.py
# tables, energy-oriented aggregation, figures, prose audit
python3 make_tables.py
python3 retuned_agg.py && python3 make_retuned_tables.py && python3 make_front_fig.py
python3 make_figs.py
python3 audit_numbers.py
```

Running `make_tables.py`, `retuned_agg.py`, `make_retuned_tables.py`, `make_front_fig.py` and `audit_numbers.py` on this repository reproduces the committed `paper/tables/*.tex` and `results/retuned_agg.json` byte for byte.

Energy-oriented roots:

```
for r in results_wos_ah results_alerce_ah results_nslkdd_ah results_wos_long results_alerce_long; do
  python3 hw_eval.py $r && cp $r/hw_eval.json $r/hw_eval_base.json && python3 ctl_eval.py $r; done
# bit-exact checks, for example
python3 check_gainp.py wos AH_F_C2_s0 1500 200 results_wos_ah
```

CPU baseline (two passes on the same machine; `make_tables.py` keeps the faster pass per run):

```
python3 cpu_bench.py results results/cpu_bench.json
python3 cpu_bench.py results results/cpu_bench_pass2.json
```

CPU energy is bounded from below with a core power of at least 1 W. `results/cpu_prev/` holds two earlier passes at other clock frequencies.

### From scratch (days)

Run the sweeps of Section 4, then the steps above.

### Where each table and figure comes from

| Paper item | Source |
|---|---|
| Table 1 (datasets) | `make_tables.py` → `datasets.tex` |
| Table 2 (accuracy-oriented hyperparameters) | `make_tables.py` → `hyper.tex`; full version in the supplement (`hyper_full.tex`) |
| Table 3 (GainP parameters) | `make_tables.py` → `params7.tex` |
| Table 4 (accuracy and model size) | `make_tables.py` → `accuracy.tex` |
| Table 5 (Mapping S, 7 nm) | `make_tables.py` → `mapS.tex` |
| Table 6 (published TM hardware) | `make_tables.py` → `tmhw.tex` (published values entered in the script from the cited papers; GainP rows computed) |
| Table 7 (energy-oriented hyperparameters) | `retuned_agg.py` + `make_retuned_tables.py` → `energyhp.tex`, `woshp.tex` |
| Fig. 1, Fig. 2 (cell and array) | TikZ in `manuscript/main.tex` |
| Fig. 3 (accuracy vs energy front) | `make_front_fig.py` |
| Supplement: Mapping C, CPU, sensitivity | `make_tables.py` → `mapC.tex`, `cpu.tex`, `sens_full.tex` (`sens.tex` is the short version) |
| Supplement: accuracy vs energy, all models | `make_figs.py` |
| Numbers in the prose | `audit_numbers.py` |

Table captions and surrounding text were edited on Overleaf after generation. The numbers in `manuscript/main.tex` are those of the generated tables.

## 7. Manuscript and submission

`manuscript/` is a snapshot of the Overleaf project (ID `6af0bb3922046c07b60edb7c`) at submission time. It compiles with `elsarticle` on Overleaf or a local TeX Live. All text and diagrams are black. Colour macros (`\hyp`, `\wosnew`, `\artem`) remain in the source as black no-ops.

`submission/` holds what was uploaded to Editorial Manager:

| File | EM item type |
|---|---|
| `Cover_letter.docx` | Cover Letter |
| `Title_page.docx` | Title Page (with author details) |
| `Manuscript_anonymized.pdf` | Manuscript (double-anonymized) |
| `Highlights.docx` | Highlights |
| `Declaration_of_competing_interest.docx` | Declaration of Interest |
| `Supplementary_material.pdf` | Supplementary Material (anonymized) |
| `CRediT_author_statement.docx` | CRediT author statement |
| `ORCID_information.pdf` | ORCID information |
| `Manuscript_FPTM_GainP.pdf` | Authored version, not uploaded; kept for reference |

Rebuild all PDFs and the Highlights file from a fresh Overleaf download:

```
python3 submission/rebuild/make_submission.py <overleaf_src_dir>     # e.g. manuscript/
python3 submission/make_tp.py <Artem's e-mail>                       # Title_page.docx, Cover_letter.docx
```

`make_submission.py` builds the authored and anonymized manuscripts and the anonymized supplement, fails on any undefined reference or citation, removes author names (author block, CRediT, a first-person sentence, the TreeTM repository entry), checks the PDF metadata, remaps the supplement's pointers into the main text ("Table N of the main text", "reference [N]") through `rebuild/old_main.aux`, and writes `Highlights.docx` and `LaTeX_source.zip`. After the main text is renumbered, refresh `old_main.aux` from the build that matches the current supplement pointers.

### Submission status (2026-11-23)

* ESWA submission ESWA-S-26-70331 is complete up to "Build PDF for Approval". All eight files are uploaded with the types above. Open access (APC) and SSRN preprint posting were selected.
* Remaining action for the corresponding author: build the PDF, approve and submit in Editorial Manager.

### Decisions made during writing

* The 7 nm figures are results of the authors' 7 nm GainP design (commercial 7 nm FinFET, PDK under NDA). The design is not fabricated. The 16 nm values are a reference.
* All text and diagrams in the submitted documents are black. Only the markers of the two data plots (Fig. 3, Fig. S1) are coloured.
* "Platform", never "substrate", for the hardware system. "Predicted" is avoided for hardware model results. The text states once that the hardware numbers come from a model.
* Funding statement left empty. AI statement: "AI was used in preparation of this work."
* `\usepackage{comment}` is loaded in the manuscript.
* Background (Section 2.1) cites 18 works from the Granmo and Yakovlev groups. Each was checked against publisher or index records (venue, year, authors).

### Optional items left open

Greyscale data plots; affiliation wording ("EnICS Labs" vs "EnICS Institute"); additions to the Scope paragraph; "modeled" vs "modelled" spelling; the roadmap phrase "evaluation methodology".

## 8. Not in this repository

* Training checkpoints (`ckpt.pkl`), per-sample prediction dumps and training logs.
* The raw WOS archive (rebuild with `prepare_data.py`).
* The GainP article drafts, evaluation workbook and review correspondence used as parameter sources. They are not ours to redistribute.
* Earlier review rounds (1 to 18) and the single-seed round results.

## 9. External references

* FPTM reference implementation: https://github.com/BooBSD/FuzzyPatternTM
* TreeTM dataset preparation: https://github.com/leonid-yavits/TreeTM
* WOS: Kowsari et al., HDLTex, 2017. ALeRCE: Sánchez-Sáez et al., AJ 2021. NSL-KDD: Tavallaee et al., CISDA 2009.
