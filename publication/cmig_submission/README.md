# CMIG Submission Package

This folder contains a standalone submission package prepared for `Computerized Medical Imaging and Graphics`.

## Contents

- `manuscript.tex`: Elsevier LaTeX manuscript in `elsarticle` format with author-year references.
- `cmig.bib`: bibliography used by the manuscript.
- `highlights.txt`: separate highlights file for Editorial Manager.
- `cover_letter.txt`: editable cover-letter draft.
- `figures/`: figure files renamed for submission.

## Compile

Run from this directory:

```bash
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

## Journal-specific items covered

- Editable LaTeX source instead of PDF-only submission.
- Abstract kept below the 250-word limit.
- Separate highlights file with 4 bullets.
- Keywords included in English.
- Numbered sections and editable tables and equations.
- Separate figure files with logical names.
- Data availability, funding, conflict of interest, ethics, and CRediT sections.
- Generative AI disclosure section placed before the references.

## Manual checks before submission

- Confirm the final corresponding author and author order in Editorial Manager.
- Confirm that each affiliation should use the listed postal address.
- Confirm that the ethics wording matches the institution's preferred formulation.
- Confirm that the public dataset and model DOIs are the final records to cite.
- Decide whether to upload any optional supplementary figures or videos.
