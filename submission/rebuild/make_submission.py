"""Rebuild the ESWA submission files from a snapshot of the Overleaf project.

Usage: python3 make_submission.py <overleaf_src_dir>
Writes latex/, anon/, supp_anon/ and the PDFs, Highlights.docx and LaTeX_source.zip in ../ (out/eswa_submission).
"""
import os, re, shutil, subprocess, sys, zipfile

SRC = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD_AUX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "old_main.aux")


def sub1(t, a, b, what):
    """Replace exactly one occurrence of a, or stop."""
    n = t.count(a)
    assert n == 1, f"{what}: found {n} occurrences"
    return t.replace(a, b)


def build(d, tex):
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex], cwd=d,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(os.path.join(d, tex.replace(".tex", ".log")), errors="ignore").read()
    und = re.findall(r"(?:Citation|Reference) `[^']+' on page \d+ undefined", log)
    assert r.returncode == 0, f"{tex} failed in {d}"
    assert not und, und
    return log


def fresh(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    shutil.copytree(os.path.join(SRC, "figures"), os.path.join(d, "figures"))


def labels(aux):
    a = open(aux).read()
    return (dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", a)),
            dict(re.findall(r"\\bibcite\{([^}]*)\}\{\{([^}]*)\}", a)))


main = open(os.path.join(SRC, "main.tex")).read()
refs = open(os.path.join(SRC, "refs.bib")).read()
supp = open(os.path.join(SRC, "supplementary.tex")).read()

# ---------------- authored version
LAT = os.path.join(OUT, "latex")
fresh(LAT)
open(os.path.join(LAT, "main.tex"), "w").write(main)
open(os.path.join(LAT, "refs.bib"), "w").write(refs)
build(LAT, "main.tex")
shutil.copy(os.path.join(LAT, "main.pdf"), os.path.join(OUT, "Manuscript_FPTM_GainP.pdf"))

# ---------------- anonymized version
an = main
m = re.search(r"\\author\[ind\].*?country=\{Israel\}\}\n", an, re.S)
assert m, "author block"
an = an.replace(m.group(0), "")
an = sub1(an, "After the main experiments, one of the authors chose new values for the FPTM and strict models of each dataset with the reference FPTM implementation",
          "After the main experiments, new values for the FPTM and strict models of each dataset were chosen with the reference FPTM implementation", "tuning sentence")
an = sub1(an, r"publicly available at \url{https://github.com/BooBSD/FuzzyPatternTM}~\cite{hnilov2025code}",
          r"publicly available~\cite{hnilov2025code}", "data availability url")
m = re.search(r"\n\\section\*\{CRediT authorship contribution statement\}.*?\n(?=\n)", an, re.S)
assert m, "credit"
an = an.replace(m.group(0), "")
m = re.search(r"\n\n\\section\*\{Funding\}\n%[^\n]*\n", an)
if m:
    an = an.replace(m.group(0), "\n")
for w in ("Hnilov, Fish", "leonid.yavits@biu", "Bar-Ilan", "EnICS"):
    assert w not in an, w
ANON = os.path.join(OUT, "anon")
fresh(ANON)
open(os.path.join(ANON, "main.tex"), "w").write(an)
m = re.search(r"@misc\{treetm2026repo,.*?\n\}\n", refs, re.S)
assert m, "treetm entry"
arefs = refs.replace(m.group(0), "@misc{treetm2026repo,\n  author       = {{Anonymous}},\n  title        = {{TreeTM} dataset preparation code},\n  howpublished = {repository link withheld for anonymized review},\n  year         = {2026}\n}\n")
open(os.path.join(ANON, "refs.bib"), "w").write(arefs)
build(ANON, "main.tex")
meta = subprocess.run(["pdfinfo", os.path.join(ANON, "main.pdf")], capture_output=True, text=True).stdout
for w in ("Hnilov", "Fish", "Yavits"):
    assert w not in meta, "PDF metadata names an author: " + w
shutil.copy(os.path.join(ANON, "main.pdf"), os.path.join(OUT, "Manuscript_anonymized.pdf"))

# ---------------- supplement (anonymized, black, pointers remapped)
oL, oC = labels(OLD_AUX)
nL, nC = labels(os.path.join(LAT, "main.aux"))
s = supp
s = re.sub(r"\\author\{[^}]*\}\n", "", s)
s = s.replace(r"\newcommand{\hyp}[1]{\textcolor{blue}{#1}}", r"\newcommand{\hyp}[1]{#1}")
s = s.replace(" Blue marks predicted values, as in the main text.", "")
s = s.replace(" (blue),", ",").replace(" (blue).", ".").replace(" Blue: predicted or assumed.", "")
assert "blue" not in s.lower().replace("fill=blue", ""), "blue left"

KIND = {"Section": "sec:", "Table": "tab:", "Fig.": "fig:", "Eq.": "eq:"}
changes = []


def remap(mt):
    kind, open_, num, close = mt.group(1), mt.group(2), mt.group(3), mt.group(4)
    cand = sorted({k for k, v in oL.items() if v == num and k.startswith(KIND[kind])} & set(nL))
    news = sorted({nL[k] for k in cand})
    assert len(news) == 1, f"{kind} {num}: {cand} -> {news}"
    changes.append(f"{kind} {num} -> {news[0]}")
    return f"{kind}~{open_}{news[0]}{close} of the main text"


s = re.sub(r"(Section|Table|Fig\.|Eq\.)~(\(?)([0-9.]+)(\)?) of the main text", remap, s)


def recite(mt):
    num = mt.group(1)
    keys = [k for k, v in oC.items() if v == num]
    assert len(keys) == 1, keys
    changes.append(f"ref {num} ({keys[0]}) -> {nC[keys[0]]}")
    return f"reference~[{nC[keys[0]]}] of the main text"


s = re.sub(r"reference~\[(\d+)\] of the main text", recite, s)
SUP = os.path.join(OUT, "supp_anon")
fresh(SUP)
open(os.path.join(SUP, "supplementary.tex"), "w").write(s)
build(SUP, "supplementary.tex")
shutil.copy(os.path.join(SUP, "supplementary.pdf"), os.path.join(OUT, "Supplementary_material.pdf"))
print("supplement pointers:", changes)

# ---------------- highlights
m = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", main, re.S)
items = [re.sub(r"\\wosnew\{(.*)\}$", r"\1", x.strip()) for x in re.findall(r"\\item\s+(.*)", m.group(1))]
items = [x.replace(r"\%", "%") for x in items]
for x in items:
    assert "\\" not in x and len(x) <= 85, x
from docx import Document
doc = Document()
doc.add_heading("Highlights", level=1)
for x in items:
    doc.add_paragraph(x, style="List Bullet")
hp = os.path.join(OUT, "Highlights.docx")
doc.save(hp)
# python-docx's default template colors headings blue: force black everywhere
with zipfile.ZipFile(hp) as zi, zipfile.ZipFile(hp + ".tmp", "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zi.infolist():
        d = zi.read(it.filename)
        if it.filename.startswith("word/") and it.filename.endswith(".xml"):
            d = re.sub(r"<w:color [^>]*/>", '<w:color w:val="000000"/>', d.decode("utf8")).encode("utf8")
        zo.writestr(it, d)
os.replace(hp + ".tmp", hp)
print("highlights:", items)

# ---------------- LaTeX source zip (authored version)
zp = os.path.join(OUT, "LaTeX_source.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("main.tex", "main.bbl", "refs.bib"):
        z.write(os.path.join(LAT, f), f)
    for f in sorted(os.listdir(os.path.join(LAT, "figures"))):
        if f.endswith(".pdf"):
            z.write(os.path.join(LAT, "figures", f), "figures/" + f)
print("done")
