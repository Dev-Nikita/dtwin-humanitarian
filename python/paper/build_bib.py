"""
Derive a BibTeX file from the manuscript reference list.

The entries follow a regular pattern, so the fields can be recovered
mechanically. The output is intended as a convenience for authors who prefer
BibTeX over the inline `thebibliography` shipped with the LaTeX package; it
should be checked against the reference list before submission.

    python -m python.paper.build_bib --out submission/latex/references.bib
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from typing import Dict

from .build_paper_ress import REFERENCES

JOURNAL_HINTS = ("IEEE", "ACM", "Journal", "Transactions", "Reliability", "Computer",
                 "Nature", "Proceedings", "Machine Learning", "Safety Science",
                 "Acta", "Engineering", "Networks", "Chaos", "PLOS", "Scandinavian",
                 "European", "Mechanical", "Earthquake", "Computing")


def _key(authors: str, year: str, index: int) -> str:
    first = re.split(r",| and ", authors.strip())[0]
    first = unicodedata.normalize("NFKD", first).encode("ascii", "ignore").decode()
    surname = first.split()[-1] if first.split() else f"anon{index}"
    return f"ref{index}"


def parse(entry: str, index: int) -> Dict[str, str]:
    """Best-effort split of one formatted reference into BibTeX fields."""
    text = entry.strip().rstrip(".")
    doi = ""
    m = re.search(r"doi:([^\s,]+)", text)
    if m:
        doi = m.group(1).rstrip(".")
        text = text[: m.start()].rstrip(" ,")

    year = ""
    # strip page ranges first: "pp. 2018-2027" must not be mistaken for a year
    ytext = re.sub(r"pp?\.\s*[\d\u2013-]+", " ", text)
    ym = re.findall(r"\b((?:19|20)\d{2})\b", ytext)
    if ym:
        year = ym[-1]

    tm = re.search(r"[“‘\"](.+?)[,.]?[”’\"]", text)
    title = tm.group(1).strip() if tm else text
    authors = text[: tm.start()].rstrip(" ,") if tm else ""
    tail = text[tm.end():].lstrip(" ,") if tm else ""

    volume = pages = number = ""
    vm = re.search(r"vol\.\s*(\d+)", tail)
    if vm:
        volume = vm.group(1)
    nm = re.search(r"no\.\s*(\d+)", tail)
    if nm:
        number = nm.group(1)
    pm = re.search(r"pp?\.\s*([\dA-Za-z]+\s*[–-]+\s*[\dA-Za-z]+|\d+)", tail)
    if pm:
        pages = pm.group(1).replace("–", "--").replace(" ", "")

    venue = tail.split(",")[0].strip() if tail else ""
    entry_type = "article"
    if venue.lower().startswith("in "):
        entry_type = "inproceedings"
        venue = venue[3:].strip()
    elif not any(h in venue for h in JOURNAL_HINTS):
        entry_type = "misc" if not venue else "article"

    authors_bib = " and ".join(a.strip() for a in re.split(r",\s*and\s*|,\s(?=[A-Z]\.)|\sand\s",
                                                           authors) if a.strip())
    return {
        "type": entry_type,
        "key": _key(authors, year, index),
        "author": authors_bib or authors,
        "title": title,
        "journal" if entry_type == "article" else "booktitle": venue,
        "volume": volume,
        "number": number,
        "pages": pages,
        "year": year,
        "doi": doi,
    }


def render(fields: Dict[str, str]) -> str:
    lines = [f"@{fields['type']}{{{fields['key']},"]
    for name in ("author", "title", "journal", "booktitle", "volume", "number",
                 "pages", "year", "doi"):
        value = fields.get(name, "")
        if value:
            lines.append(f"  {name:9s} = {{{value}}},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("submission", "latex", "references.bib"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    blocks = [render(parse(ref, i)) for i, ref in enumerate(REFERENCES, start=1)]
    header = ("% Auto-derived from the manuscript reference list.\n"
              "% Verify fields before using with a .bst style; the LaTeX package\n"
              "% ships a self-contained thebibliography that needs no BibTeX run.\n\n")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(header + "\n\n".join(blocks) + "\n")
    print("bib:", args.out, f"({len(blocks)} entries)")


if __name__ == "__main__":
    main()
