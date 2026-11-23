#!/bin/bash
# Two-pass build of main.tex and supplementary.tex (cross-document numbers). Usage: build_all.sh <scratch dir>
set -e
E=$1; mkdir -p $E/figures
cd "$(dirname "$0")"
for pass in 1 2; do
  python3 make_tables.py >/dev/null 2>&1
  python3 assemble.py >/dev/null
  cp out/overleaf/main.tex out/overleaf/refs.bib $E/; cp out/overleaf/figures/*.pdf $E/figures/
  (cd $E && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 || true)
  MAIN_AUX=$E/main.aux python3 build_supp.py >/dev/null
  cp out/overleaf/supplementary.tex $E/
  (cd $E && latexmk -pdf -interaction=nonstopmode supplementary.tex >/dev/null 2>&1 || true)
  python3 build_supp.py --labels $E/supplementary.aux >/dev/null
done
python3 assemble.py >/dev/null; cmp -s out/overleaf/main.tex $E/main.tex && echo "main stable" || echo "MAIN CHANGED"
echo "main pages: $(pdfinfo $E/main.pdf | grep Pages | awk '{print $2}'), supp pages: $(pdfinfo $E/supplementary.pdf | grep Pages | awk '{print $2}')"
grep -c Overfull $E/main.log || true; grep -c Overfull $E/supplementary.log || true
grep -n "Undefined control\|undefined\|Error" $E/main.log $E/supplementary.log | head
cat out/overleaf/supp_labels.json | tr '\n' ' '; echo
grep -o "S?" out/overleaf/main.tex | wc -l
