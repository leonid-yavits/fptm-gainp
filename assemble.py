"""Assemble the single-file Overleaf project (author preference: all text in
one main.tex, no \\input of section or table files)."""
import json, os, re, shutil, sys
SRC = "paper"
OUT = sys.argv[1] if len(sys.argv) > 1 else "out/overleaf"


def inline(path):
    txt = open(path).read()
    def rep(m):
        p = m.group(1)
        p = p if p.endswith(".tex") else p + ".tex"
        return inline(os.path.join(SRC, p)).rstrip("\n")
    return re.sub(r"\\input\{([^}]*)\}", rep, txt)


os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
main = inline(os.path.join(SRC, "main.tex"))
# supplementary-material numbers (S1, Table S2, ...) come from the compiled supplement (build_supp.py -> supp_labels.json)
_SL = json.load(open(os.path.join(OUT, "supp_labels.json"))) if os.path.exists(os.path.join(OUT, "supp_labels.json")) else {}
main = re.sub(r"\\supref\{([^}]*)\}", lambda m: _SL.get(m.group(1), "S?"), main)
assert "\\input{" not in main
open(os.path.join(OUT, "main.tex"), "w").write(main)
shutil.copy(os.path.join(SRC, "refs.bib"), OUT)
for f in os.listdir(os.path.join(SRC, "figures")):
    if f.endswith(".pdf"):
        shutil.copy(os.path.join(SRC, "figures", f), os.path.join(OUT, "figures", f))
print("assembled", os.path.join(OUT, "main.tex"), len(main.splitlines()), "lines")
