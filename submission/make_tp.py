import sys
from docx import Document
from docx.shared import Pt
import os; S=os.path.dirname(os.path.abspath(__file__))+'/'
AE=sys.argv[1]
def doc():
    d=Document(); st=d.styles['Normal']; st.font.name='Times New Roman'; st.font.size=Pt(12); return d
d=doc(); d.add_heading('Title page',level=1)
p=d.add_paragraph(); p.add_run('Fuzzy-Pattern Tsetlin Machine Inference on a Gain Cell Embedded DRAM Associative In-Memory Processor').bold=True
d.add_paragraph('Artem Hnilov (a), Alexander Fish (b), Leonid Yavits (b, *)')
d.add_paragraph('(a) Independent Researcher, France\n(b) EnICS Labs, Faculty of Engineering, Bar-Ilan University, Ramat Gan 5290002, Israel')
d.add_paragraph('E-mail addresses: Artem Hnilov, %s; Alexander Fish, alexander.fish@biu.ac.il; Leonid Yavits, leonid.yavits@biu.ac.il' % AE)
d.add_paragraph('(*) Corresponding author: Leonid Yavits, EnICS Labs, Faculty of Engineering, Bar-Ilan University, Ramat Gan 5290002, Israel. E-mail: leonid.yavits@biu.ac.il')
d.add_heading('CRediT authorship contribution statement',level=2)
d.add_paragraph('Artem Hnilov: Software, Data curation, Methodology, Validation, Formal analysis. Alexander Fish: Conceptualization, Visualization. Leonid Yavits: Conceptualization, Writing – original draft, Writing – review & editing, Supervision, Project administration, Resources.')
d.add_heading('Data and code availability',level=2)
d.add_paragraph('The FPTM reference implementation is publicly available at https://github.com/BooBSD/FuzzyPatternTM. The datasets are publicly available from their original sources. The ALeRCE and NSL-KDD splits follow the published TreeTM preparation code at https://github.com/leonid-yavits/TreeTM.')
d.add_heading('Declaration of competing interest',level=2)
d.add_paragraph('The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.')
d.add_heading('Declaration of generative AI and AI-assisted technologies in the manuscript preparation process',level=2)
d.add_paragraph('AI was used in preparation of this work.')
d.save(S+'Title_page.docx')
# cover letter
d=doc()
for p in ["Leonid Yavits\nEnICS Labs, Faculty of Engineering\nBar-Ilan University, Ramat Gan, Israel\nleonid.yavits@biu.ac.il",
"To the Editor-in-Chief,\nExpert Systems with Applications",
"Dear Editor,",
"We submit the manuscript “Fuzzy-Pattern Tsetlin Machine Inference on a Gain Cell Embedded DRAM Associative In-Memory Processor” by Artem Hnilov, Alexander Fish and Leonid Yavits for consideration as a full length article in Expert Systems with Applications.",
"Tsetlin machines are interpretable, rule-based classifiers. Each class is decided by a vote of propositional clauses that a domain expert can read. The Fuzzy-Pattern Tsetlin Machine (FPTM) lets a clause vote with a reduced score when some of its literals fail, and it needs far fewer clauses than the classic model. This paper asks whether this advantage survives when the rule base runs inside an associative in-memory processor, a hardware target that suits low-energy edge and streaming applications of expert systems.",
"Novel contributions. (1) A mapping of FPTM inference onto an associative in-memory processor, GainP, in a sample-stationary and a clause-stationary layout, with exact compile-time clause simplification and two in-array miss counters. (2) A functional simulator that confirms bit-exact agreement of the in-array predictions with a software reference. (3) An evaluation of FPTM, strict and vanilla Tsetlin machine models on text classification (WOS-46985), astronomical alert classification (ALeRCE) and network intrusion detection (NSL-KDD), with costs evaluated for a 7 nm GainP design and compared with published Tsetlin machine ASIC, FPGA and in-memory accelerators. (4) The finding that the hardware cost of FPTM follows its total number of included literals rather than its clause count: with hyperparameters tuned for few included literals, FPTM models on NSL-KDD reach the accuracy band of the best strict model at 13x to 23x less energy. The paper also states where FPTM does not pay off, and gives practitioners a rule for choosing between the two rule bases.",
"Author e-mail addresses. Alexander Fish and Leonid Yavits use their institutional addresses at Bar-Ilan University. Artem Hnilov is an independent researcher and the author of the reference FPTM implementation. He has no institutional affiliation, and therefore no institutional e-mail address. His personal address, %s, is given in the title page and in the submission system." % AE,
"The manuscript is original. It has not been published and is not under consideration for publication elsewhere. All authors have approved the manuscript and agree with its submission. Supplementary material accompanies the manuscript. The authors declare no competing interests.",
"Thank you for considering our work.",
"Sincerely,\nLeonid Yavits, on behalf of all authors"]:
    d.add_paragraph(p)
d.save(S+'Cover_letter.docx')
