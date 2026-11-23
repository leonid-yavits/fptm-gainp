"""Local compile check of out/overleaf/main.tex with an article-class shim
(elsarticle.cls is not installed here; Overleaf provides it)."""
import re, subprocess, os
src = open("out/overleaf/main.tex").read()
s = src.replace(r"\documentclass[preprint,review,12pt]{elsarticle}", r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\newenvironment{frontmatter}{}{}
\newenvironment{highlights}{\section*{Highlights}\begin{itemize}}{\end{itemize}}
\newenvironment{keyword}{\noindent\textbf{Keywords:} }{}
\newcommand{\sep}{, }
\newcommand{\ead}[1]{}
\newcommand{\cortext}[2][]{}
\newcommand{\corref}[1]{}
\newcommand{\affiliation}[2][]{}
\newcommand{\journal}[1]{}""")
s = re.sub(r"\\author\[[^\]]*\]\{", r"\\author{", s)
s = s.replace(r"\bibliographystyle{elsarticle-num}", r"\bibliographystyle{unsrt}")
s = s.replace(r"\begin{abstract}", r"\maketitle\begin{abstract}", 1)
open("out/overleaf/check_build.tex", "w").write(s)
r = subprocess.run("cd out/overleaf && rm -f check_build.aux check_build.bbl check_build.pdf check_build.fdb_latexmk && latexmk -pdf -interaction=nonstopmode check_build.tex > build.log 2>&1", shell=True)
log = open("out/overleaf/check_build.log").read()
errs = [l for l in log.splitlines() if l.startswith("!") or "undefined" in l]
print("errors/undefined:", errs[:15])
print(subprocess.run("pdfinfo out/overleaf/check_build.pdf | grep Pages", shell=True, capture_output=True, text=True).stdout)
