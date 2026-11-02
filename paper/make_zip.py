"""Pack the arXiv upload: main.tex and the figure PDFs, nothing else.

    python paper/make_zip.py
"""
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["main.tex", *sorted(f"figures/{p.name}" for p in (HERE / "figures").glob("*.pdf"))]

with zipfile.ZipFile(HERE / "arxiv_submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name in FILES:
        z.write(HERE / name, name)
print("wrote paper/arxiv_submission.zip:", ", ".join(FILES))
