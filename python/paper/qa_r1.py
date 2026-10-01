"""
Consistency audit of the R1 submission package.

    python -m python.paper.qa_r1 --docx X.docx --pkg DIR

1. No legacy claim of the first submission survives in the manuscript,
   highlights or cover letter.
2. Every headline number in the abstract, conclusion, highlights, cover letter
   and response letter is recomputed from data/results and must appear verbatim.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd
from docx import Document

BANNED = [
    r"\b33 ?%", r"\b74 ?%", r"two thirds", r"over-?actuates", r"external validation",
    r"all policies are indistinguishable", r"worse than not acting",
    r"prediction drives", r"first version", r"first submission", r"Reviewer \d",
    r"is the best policy", r"the best policy at every", r"optimal policy", r"\bgeneralis(e|es|ed)\b",
    r"never worse", r"never hurts", r"stays behind plain protection",
    r"does better than predictive migration", r"two per cent of one another",
    r"previously reported", r"anticipation drives", r"pre-?registered",
    r"does not hold", r"pays most reliably through protection",
]
# phrases allowed when negated / quoted as limitation
ALLOW = [r"not an external validation", r"not external validation",
         r"no claim is made", r"rather than an external validation"]


def text_of(path):
    if path.endswith(".docx"):
        return "\n".join(p.text for p in Document(path).paragraphs) + "\n" + "\n".join(
            c.text for t in Document(path).tables for r in t.rows for c in r.cells)
    return open(path, encoding="utf-8").read()


def scan(name, text, banned):
    bad = []
    for pat in banned:
        for m in re.finditer(pat, text, flags=re.I):
            ctx = text[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")
            if any(re.search(a, ctx, flags=re.I) for a in ALLOW):
                continue
            bad.append(f"{name}: '{m.group(0)}' … {ctx}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True)
    ap.add_argument("--pkg", required=True)
    a = ap.parse_args()
    problems = []
    man = text_of(a.docx)
    body = man.split("\nReferences\n")[0]
    problems += scan("manuscript", body, BANNED)
    for f in ("Highlights_R1.txt", "Cover_Letter_R1.md"):
        p = os.path.join(a.pkg, f)
        if os.path.exists(p):
            problems += scan(f, text_of(p), [b for b in BANNED
                                             if b not in (r"first submission", r"first version",
                                                          r"\b33 ?%", r"Reviewer \d")])
    # headline numbers
    from .build_response_r1 import headline_numbers
    nums = headline_numbers()
    for label, value in nums.items():
        for f, txt in (("manuscript", man),
                       ("response", text_of(os.path.join(a.pkg, "Response_to_Reviewers.md")))):
            if label.startswith(f + ":") and value not in txt:
                problems.append(f"{f}: headline number {label} = {value} not found")
    print("\n".join(problems) if problems else "QA: all checks passed")
    print(f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
