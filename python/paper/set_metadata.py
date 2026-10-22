"""Set author and title metadata on the generated Word documents.

    python -m python.paper.set_metadata DIR
"""
import datetime as _dt
import os
import sys

from docx import Document

AUTHOR = "Nikita Tarasov"
TITLES = {
    "Manuscript_R1.docx": "Where Does Predictive Resilience Come From? (revised manuscript)",
    "Response_to_Reviewers.docx": "JRESS-D-26-06464 - response to the reviewers",
    "Cover_Letter_R1.docx": "JRESS-D-26-06464 - cover letter",
    "Highlights_R1.docx": "JRESS-D-26-06464 - highlights",
}


def main(folder: str) -> None:
    now = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
    for name, title in TITLES.items():
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        doc = Document(path)
        cp = doc.core_properties
        cp.author = AUTHOR
        cp.last_modified_by = AUTHOR
        cp.title = title
        cp.comments = ""
        cp.keywords = ""
        cp.created = now
        cp.modified = now
        doc.save(path)
        print("metadata:", name)


if __name__ == "__main__":
    main(sys.argv[1])
