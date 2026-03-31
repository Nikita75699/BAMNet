from __future__ import annotations

import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path("/mnt/ssd4tb/project/BAMNet")
CMIG_DIR = REPO_ROOT / "publication" / "cmig_submission"
TEX_PATH = CMIG_DIR / "manuscript.tex"
BIB_PATH = CMIG_DIR / "cmig.bib"
OUTPUT_PATH = CMIG_DIR / "manuscript_controlled.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
O_NS = "urn:schemas-microsoft-com:office:office"
V_NS = "urn:schemas-microsoft-com:vml"
W10_NS = "http://schemas.microsoft.com/office/word/2003/wordml"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"

NS = {
    "w": W_NS,
    "m": M_NS,
    "r": R_NS,
    "o": O_NS,
    "v": V_NS,
    "w10": W10_NS,
    "a": A_NS,
    "pic": PIC_NS,
    "wp": WP_NS,
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


FIGURE_MAP = {
    "fig:annotation_examples": "Figure 1",
    "fig:bamnet_architecture": "Figure 2",
    "fig:bamnet_qualitative": "Figure 3",
    "fig:ablation_summary": "Figure 4",
}

TABLE_MAP = {
    "tab:seg_baselines": "Table 1",
    "tab:landmark_baselines": "Table 2",
    "tab:crossval": "Table 3",
    "tab:landmark_summary": "Table 4",
    "tab:landmark_folds": "Table 5",
    "tab:ablation_results": "Table 6",
}

SECTION_MAP = {
    "subsec:baseline_results": "Section 3.1",
    "subsec:crossval": "Section 3.3",
}

AFFILIATIONS = [
    "¹ Siberian State Medical University, 2 Moskovsky Trakt, Tomsk 634050, Russia",
    "² V.A. Trapeznikov Institute of Control Sciences of Russian Academy of Sciences, 65 Profsoyuznaya Street, Moscow 117997, Russia",
    "³ Almazov National Medical Research Centre, 2 Akkuratova Street, Saint Petersburg 197341, Russia",
    "⁴ Pompeu Fabra University, Carrer de la Mercè 12, 08002 Barcelona, Spain",
]


def qn(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def find_block(text: str, start_token: str, end_token: str) -> str:
    start = text.index(start_token) + len(start_token)
    end = text.index(end_token, start)
    return text[start:end]


def extract_brace_arg(text: str, token: str, start_pos: int = 0) -> tuple[str, int]:
    pos = text.index(token, start_pos)
    i = pos + len(token)
    depth = 1
    out: list[str] = []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out), i


def latex_accents(text: str) -> str:
    text = text.replace(r"\`{e}", "è")
    text = text.replace(r"\'{e}", "é")
    text = text.replace(r"\"{o}", "ö")
    text = text.replace(r"\"{u}", "ü")
    text = text.replace(r"\"{a}", "ä")
    text = text.replace(r"\~{n}", "ñ")
    text = text.replace(r"\c{c}", "ç")
    text = text.replace(r"\'{i}", "í")
    text = text.replace(r"\'{a}", "á")
    text = text.replace(r"\'{o}", "ó")
    text = text.replace(r"\'{u}", "ú")
    text = text.replace(r"\`{a}", "à")
    text = text.replace(r"\`{i}", "ì")
    text = text.replace(r"\`{o}", "ò")
    text = text.replace(r"\`{u}", "ù")
    return text


def convert_inline_latex(text: str, keep_citations: bool = True) -> str:
    text = latex_accents(text)
    text = text.replace(r"\&", "&")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\_", "_")
    text = text.replace(r"\#", "#")
    text = text.replace(r"~", " ")
    text = text.replace(r"``", '"')
    text = text.replace(r"''", '"')
    text = re.sub(r"\\textbf\{([^{}]+)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{([^{}]+)\}", r"*\1*", text)
    text = re.sub(r"\\emph\{([^{}]+)\}", r"*\1*", text)
    text = re.sub(r"\\url\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\href\{([^{}]+)\}\{([^{}]+)\}", r"[\2](\1)", text)
    text = re.sub(r"\\citep?\{([^{}]+)\}", cite_to_pandoc if keep_citations else "", text)
    text = re.sub(r"\\label\{[^{}]+\}", "", text)
    text = re.sub(r"\\(noindent|centering|hfill)\b", "", text)
    return text


def cite_to_pandoc(match: re.Match[str]) -> str:
    keys = [key.strip() for key in match.group(1).split(",")]
    if not keys:
        return ""
    return "[" + "; ".join(f"@{key}" for key in keys) + "]"


def resolve_refs(text: str) -> str:
    for key, val in FIGURE_MAP.items():
        text = text.replace(fr"Fig.~\ref{{{key}}}", val)
        text = text.replace(fr"Figure~\ref{{{key}}}", val)
    for key, val in TABLE_MAP.items():
        text = text.replace(fr"Table~\ref{{{key}}}", val)
    for key, val in SECTION_MAP.items():
        text = text.replace(fr"Section~\ref{{{key}}}", val)
    text = re.sub(r"\\ref\{[^{}]+\}", "REF", text)
    return text


def convert_equations(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        return f"\n\n$$\n{body}\n$$\n\n"

    return re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", repl, text, flags=re.S)


def convert_lists(text: str) -> str:
    def enum_repl(match: re.Match[str]) -> str:
        items = re.findall(r"\\item\s+(.*?)(?=(\\item|$))", match.group(1), flags=re.S)
        out = []
        for idx, (item, _) in enumerate(items, 1):
            out.append(f"{idx}. {convert_inline_latex(item)}")
        return "\n\n" + "\n".join(out) + "\n\n"

    def item_repl(match: re.Match[str]) -> str:
        items = re.findall(r"\\item\s+(.*?)(?=(\\item|$))", match.group(1), flags=re.S)
        out = [f"- {convert_inline_latex(item)}" for item, _ in items]
        return "\n\n" + "\n".join(out) + "\n\n"

    text = re.sub(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", enum_repl, text, flags=re.S)
    text = re.sub(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", item_repl, text, flags=re.S)
    return text


def strip_table_wrappers(text: str) -> str:
    text = re.sub(r"\\resizebox\{[^{}]+\}\{[^{}]+\}\{%?", "", text)
    text = text.replace("}}", "}\n")
    return text


def latex_cell_to_text(cell: str) -> str:
    cell = cell.strip()
    cell = convert_inline_latex(cell, keep_citations=False)
    cell = cell.replace("{", "").replace("}", "")
    cell = cell.replace("\\", "")
    return cell.strip()


def parse_tabular_rows(tabular_body: str) -> list[list[str]]:
    body = strip_comments(tabular_body)
    body = body.replace(r"\toprule", "")
    body = body.replace(r"\midrule", "")
    body = body.replace(r"\bottomrule", "")
    body = body.replace(r"\,", " ")
    rows: list[list[str]] = []
    for raw_row in re.split(r"\\\\", body):
        row = raw_row.strip()
        if not row:
            continue
        cells = [latex_cell_to_text(cell) for cell in row.split("&")]
        if any(cells):
            rows.append(cells)
    return rows


def table_env_to_markdown(match: re.Match[str]) -> str:
    block = match.group(0)
    caption, _ = extract_brace_arg(block, r"\caption{")
    label_match = re.search(r"\\label\{([^{}]+)\}", block)
    label = label_match.group(1) if label_match else ""
    tabular_match = re.search(r"\\begin\{tabular\}\{([^{}]+)\}(.*?)\\end\{tabular\}", block, flags=re.S)
    if not tabular_match:
        return ""
    rows = parse_tabular_rows(tabular_match.group(2))
    if not rows:
        return ""
    header = rows[0]
    align = "|" + "|".join(["---"] * len(header)) + "|"
    lines = [
        f"**{TABLE_MAP.get(label, 'Table')}. {convert_inline_latex(caption, keep_citations=False)}**",
        "",
        "|" + "|".join(header) + "|",
        align,
    ]
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append("|" + "|".join(padded[: len(header)]) + "|")
    return "\n\n" + "\n".join(lines) + "\n\n"


def combine_images(image_paths: list[Path], out_path: Path, cols: int, labels: list[str]) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    gap = 40
    rows = (len(images) + cols - 1) // cols
    canvas_w = cols * max_w + (cols - 1) * gap
    canvas_h = rows * max_h + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * (max_w + gap)
        y = row * (max_h + gap)
        if img.width != max_w or img.height != max_h:
            scaled = img.copy()
            scaled.thumbnail((max_w, max_h))
            paste_x = x + (max_w - scaled.width) // 2
            paste_y = y + (max_h - scaled.height) // 2
            canvas.paste(scaled, (paste_x, paste_y))
        else:
            canvas.paste(img, (x, y))
        if idx < len(labels):
            draw.rectangle((x + 8, y + 8, x + 48, y + 28), fill="white")
            draw.text((x + 14, y + 11), labels[idx], fill="black", font=font)

    canvas.save(out_path)


def figure_block_markdown(name: str, image_name: str, caption: str) -> str:
    return f"\n\n![{name}. {caption}]({image_name}){{ width=95% }}\n\n"


def normalize_markdown(text: str) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    return "\n".join(lines).strip() + "\n"


def escape_metric_at(text: str) -> str:
    replacements = {
        "Surface Dice@4 mm": "Surface Dice at 4 mm",
        "PCK@10": "PCK at 10",
        "PCK@r": "PCK at r",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def find_style(styles_root: ET.Element, style_id: str) -> ET.Element | None:
    for style in styles_root.findall(qn("w", "style")):
        if style.get(qn("w", "styleId")) == style_id:
            return style
    return None


def set_val_child(parent: ET.Element, tag: str, value: str) -> ET.Element:
    child = parent.find(qn("w", tag))
    if child is None:
        child = ET.SubElement(parent, qn("w", tag))
    child.set(qn("w", "val"), value)
    return child


def remove_child(parent: ET.Element, tag: str) -> None:
    child = parent.find(qn("w", tag))
    if child is not None:
        parent.remove(child)


def reset_child(parent: ET.Element, tag: str) -> ET.Element:
    remove_child(parent, tag)
    return ET.SubElement(parent, qn("w", tag))


def ensure_style(
    styles_root: ET.Element,
    style_id: str,
    name: str,
    *,
    style_type: str = "paragraph",
    based_on: str | None = None,
    next_style: str | None = None,
    custom: bool = False,
) -> ET.Element:
    style = find_style(styles_root, style_id)
    if style is None:
        attrs = {
            qn("w", "styleId"): style_id,
            qn("w", "type"): style_type,
        }
        if custom:
            attrs[qn("w", "customStyle")] = "1"
        style = ET.SubElement(styles_root, qn("w", "style"), attrs)
    else:
        style.set(qn("w", "styleId"), style_id)
        style.set(qn("w", "type"), style_type)
        if custom:
            style.set(qn("w", "customStyle"), "1")
    set_val_child(style, "name", name)
    if based_on is not None:
        set_val_child(style, "basedOn", based_on)
    if next_style is not None:
        set_val_child(style, "next", next_style)
    if style.find(qn("w", "qFormat")) is None:
        ET.SubElement(style, qn("w", "qFormat"))
    return style


def set_style_run_props(
    style: ET.Element,
    *,
    size: int,
    bold: bool = False,
    italic: bool = False,
    color: str = "000000",
) -> None:
    r_pr = reset_child(style, "rPr")
    ET.SubElement(
        r_pr,
        qn("w", "rFonts"),
        {
            qn("w", "ascii"): "Times New Roman",
            qn("w", "hAnsi"): "Times New Roman",
            qn("w", "cs"): "Times New Roman",
        },
    )
    if bold:
        ET.SubElement(r_pr, qn("w", "b"))
        ET.SubElement(r_pr, qn("w", "bCs"))
    if italic:
        ET.SubElement(r_pr, qn("w", "i"))
        ET.SubElement(r_pr, qn("w", "iCs"))
    ET.SubElement(r_pr, qn("w", "color"), {qn("w", "val"): color})
    ET.SubElement(r_pr, qn("w", "sz"), {qn("w", "val"): str(size)})
    ET.SubElement(r_pr, qn("w", "szCs"), {qn("w", "val"): str(size)})


def set_style_paragraph_props(
    style: ET.Element,
    *,
    before: int = 0,
    after: int = 0,
    line: int | None = None,
    line_rule: str | None = None,
    justification: str | None = None,
    keep_next: bool = False,
    keep_lines: bool = False,
    outline_level: int | None = None,
    left: int | None = None,
    hanging: int | None = None,
) -> None:
    p_pr = reset_child(style, "pPr")
    if keep_next:
        ET.SubElement(p_pr, qn("w", "keepNext"))
    if keep_lines:
        ET.SubElement(p_pr, qn("w", "keepLines"))
    spacing_attrs = {qn("w", "before"): str(before), qn("w", "after"): str(after)}
    if line is not None:
        spacing_attrs[qn("w", "line")] = str(line)
    if line_rule is not None:
        spacing_attrs[qn("w", "lineRule")] = line_rule
    ET.SubElement(p_pr, qn("w", "spacing"), spacing_attrs)
    if justification is not None:
        ET.SubElement(p_pr, qn("w", "jc"), {qn("w", "val"): justification})
    if left is not None or hanging is not None:
        ind_attrs: dict[str, str] = {}
        if left is not None:
            ind_attrs[qn("w", "left")] = str(left)
        if hanging is not None:
            ind_attrs[qn("w", "hanging")] = str(hanging)
        ET.SubElement(p_pr, qn("w", "ind"), ind_attrs)
    if outline_level is not None:
        ET.SubElement(p_pr, qn("w", "outlineLvl"), {qn("w", "val"): str(outline_level)})


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", NS)).strip()


def set_paragraph_style(paragraph: ET.Element, style_id: str) -> None:
    p_pr = paragraph.find(qn("w", "pPr"))
    if p_pr is None:
        p_pr = ET.Element(qn("w", "pPr"))
        paragraph.insert(0, p_pr)
    p_style = p_pr.find(qn("w", "pStyle"))
    if p_style is None:
        p_style = ET.Element(qn("w", "pStyle"))
        p_pr.insert(0, p_style)
    p_style.set(qn("w", "val"), style_id)


def remove_direct_bold(paragraph: ET.Element) -> None:
    for r_pr in paragraph.findall(".//w:rPr", NS):
        for tag in ("b", "bCs"):
            child = r_pr.find(qn("w", tag))
            if child is not None:
                r_pr.remove(child)


def make_text_paragraph(text: str, style_id: str) -> ET.Element:
    paragraph = ET.Element(qn("w", "p"))
    p_pr = ET.SubElement(paragraph, qn("w", "pPr"))
    ET.SubElement(p_pr, qn("w", "pStyle"), {qn("w", "val"): style_id})
    run = ET.SubElement(paragraph, qn("w", "r"))
    text_node = ET.SubElement(run, qn("w", "t"))
    text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text
    return paragraph


def polish_styles(styles_root: ET.Element) -> None:
    doc_defaults = styles_root.find(qn("w", "docDefaults"))
    if doc_defaults is not None:
        r_pr_default = doc_defaults.find(f"{qn('w', 'rPrDefault')}/{qn('w', 'rPr')}")
        if r_pr_default is not None:
            for child in list(r_pr_default):
                r_pr_default.remove(child)
            ET.SubElement(
                r_pr_default,
                qn("w", "rFonts"),
                {
                    qn("w", "ascii"): "Times New Roman",
                    qn("w", "hAnsi"): "Times New Roman",
                    qn("w", "cs"): "Times New Roman",
                },
            )
            ET.SubElement(r_pr_default, qn("w", "sz"), {qn("w", "val"): "22"})
            ET.SubElement(r_pr_default, qn("w", "szCs"), {qn("w", "val"): "22"})
            ET.SubElement(
                r_pr_default,
                qn("w", "lang"),
                {
                    qn("w", "val"): "en-US",
                    qn("w", "eastAsia"): "en-US",
                    qn("w", "bidi"): "ar-SA",
                },
            )
        p_pr_default = doc_defaults.find(f"{qn('w', 'pPrDefault')}/{qn('w', 'pPr')}")
        if p_pr_default is not None:
            for child in list(p_pr_default):
                p_pr_default.remove(child)
            ET.SubElement(
                p_pr_default,
                qn("w", "spacing"),
                {
                    qn("w", "before"): "0",
                    qn("w", "after"): "120",
                    qn("w", "line"): "276",
                    qn("w", "lineRule"): "auto",
                },
            )

    normal = ensure_style(styles_root, "Normal", "Normal")
    set_style_paragraph_props(normal, before=0, after=120, line=276, line_rule="auto")
    set_style_run_props(normal, size=22)

    body = ensure_style(styles_root, "BodyText", "Body Text", based_on="Normal", next_style="BodyText")
    set_style_paragraph_props(body, before=0, after=120, line=276, line_rule="auto")
    set_style_run_props(body, size=22)

    first = ensure_style(
        styles_root,
        "FirstParagraph",
        "First Paragraph",
        based_on="BodyText",
        next_style="BodyText",
        custom=True,
    )
    set_style_paragraph_props(first, before=0, after=120, line=276, line_rule="auto")
    set_style_run_props(first, size=22)

    compact = ensure_style(styles_root, "Compact", "Compact", based_on="BodyText", custom=True)
    set_style_paragraph_props(compact, before=0, after=24, line=240, line_rule="auto")
    set_style_run_props(compact, size=20)

    bibliography = ensure_style(
        styles_root,
        "Bibliography",
        "Bibliography",
        based_on="Normal",
        next_style="Bibliography",
    )
    set_style_paragraph_props(
        bibliography,
        before=0,
        after=80,
        line=240,
        line_rule="auto",
        left=360,
        hanging=360,
    )
    set_style_run_props(bibliography, size=20)

    title = ensure_style(styles_root, "Title", "Title", based_on="Normal", next_style="Author")
    set_style_paragraph_props(title, before=240, after=180, justification="center", keep_next=True, keep_lines=True)
    set_style_run_props(title, size=30, bold=True)

    author = ensure_style(styles_root, "Author", "Author", next_style="Affiliation", custom=True)
    set_style_paragraph_props(author, before=60, after=40, justification="center", keep_next=True, keep_lines=True)
    set_style_run_props(author, size=22)

    affiliation = ensure_style(
        styles_root,
        "Affiliation",
        "Affiliation",
        based_on="BodyText",
        next_style="Correspondence",
        custom=True,
    )
    set_style_paragraph_props(affiliation, before=0, after=40, justification="center", line=240, line_rule="auto")
    set_style_run_props(affiliation, size=20)

    correspondence = ensure_style(
        styles_root,
        "Correspondence",
        "Correspondence",
        based_on="BodyText",
        next_style="AbstractTitle",
        custom=True,
    )
    set_style_paragraph_props(correspondence, before=0, after=160, justification="center", line=240, line_rule="auto")
    set_style_run_props(correspondence, size=20)

    abstract_title = ensure_style(
        styles_root,
        "AbstractTitle",
        "Abstract Title",
        based_on="Normal",
        next_style="Abstract",
        custom=True,
    )
    set_style_paragraph_props(
        abstract_title,
        before=180,
        after=40,
        justification="center",
        keep_next=True,
        keep_lines=True,
    )
    set_style_run_props(abstract_title, size=22, bold=True)

    abstract = ensure_style(
        styles_root,
        "Abstract",
        "Abstract",
        based_on="BodyText",
        next_style="Keywords",
        custom=True,
    )
    set_style_paragraph_props(abstract, before=0, after=80, line=276, line_rule="auto")
    set_style_run_props(abstract, size=22)

    keywords = ensure_style(
        styles_root,
        "Keywords",
        "Keywords",
        based_on="BodyText",
        next_style="Heading1",
        custom=True,
    )
    set_style_paragraph_props(keywords, before=0, after=160, line=240, line_rule="auto")
    set_style_run_props(keywords, size=21)

    heading1 = ensure_style(styles_root, "Heading1", "Heading 1", based_on="Normal", next_style="BodyText")
    set_style_paragraph_props(
        heading1,
        before=240,
        after=40,
        keep_next=True,
        keep_lines=True,
        outline_level=0,
    )
    set_style_run_props(heading1, size=24, bold=True)

    heading2 = ensure_style(styles_root, "Heading2", "Heading 2", based_on="Normal", next_style="BodyText")
    set_style_paragraph_props(
        heading2,
        before=180,
        after=20,
        keep_next=True,
        keep_lines=True,
        outline_level=1,
    )
    set_style_run_props(heading2, size=22, bold=True)

    caption = ensure_style(styles_root, "Caption", "Caption", based_on="Normal", next_style="BodyText")
    set_style_paragraph_props(caption, before=60, after=60, line=240, line_rule="auto")
    set_style_run_props(caption, size=20)

    table_caption = ensure_style(
        styles_root,
        "TableCaption",
        "Table Caption",
        based_on="Caption",
        next_style="TableCaption",
        custom=True,
    )
    set_style_paragraph_props(table_caption, before=120, after=40, keep_next=True)
    set_style_run_props(table_caption, size=20)

    image_caption = ensure_style(
        styles_root,
        "ImageCaption",
        "Image Caption",
        based_on="Caption",
        next_style="BodyText",
        custom=True,
    )
    set_style_paragraph_props(image_caption, before=20, after=100, justification="center", line=240, line_rule="auto")
    set_style_run_props(image_caption, size=20)

    figure = ensure_style(styles_root, "Figure", "Figure", based_on="Normal", custom=True)
    set_style_paragraph_props(figure, before=80, after=20, justification="center")
    set_style_run_props(figure, size=20)

    captioned_figure = ensure_style(
        styles_root,
        "CaptionedFigure",
        "Captioned Figure",
        based_on="Figure",
        next_style="ImageCaption",
        custom=True,
    )
    set_style_paragraph_props(captioned_figure, before=80, after=20, justification="center", keep_next=True)
    set_style_run_props(captioned_figure, size=20)

    table_style = ensure_style(styles_root, "Table", "Table", style_type="table", based_on="TableNormal")
    tbl_pr = reset_child(table_style, "tblPr")
    ET.SubElement(tbl_pr, qn("w", "tblInd"), {qn("w", "type"): "dxa", qn("w", "w"): "0"})
    ET.SubElement(tbl_pr, qn("w", "jc"), {qn("w", "val"): "center"})
    borders = ET.SubElement(tbl_pr, qn("w", "tblBorders"))
    for border_name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        ET.SubElement(
            borders,
            qn("w", border_name),
            {
                qn("w", "val"): "single",
                qn("w", "sz"): "4",
                qn("w", "space"): "0",
                qn("w", "color"): "808080",
            },
        )
    cell_mar = ET.SubElement(tbl_pr, qn("w", "tblCellMar"))
    for side, width in (("top", "60"), ("left", "120"), ("bottom", "60"), ("right", "120")):
        ET.SubElement(cell_mar, qn("w", side), {qn("w", "type"): "dxa", qn("w", "w"): width})

    for tbl_style_pr in list(table_style.findall(qn("w", "tblStylePr"))):
        table_style.remove(tbl_style_pr)
    first_row = ET.SubElement(table_style, qn("w", "tblStylePr"), {qn("w", "type"): "firstRow"})
    first_row_tc = ET.SubElement(first_row, qn("w", "tcPr"))
    ET.SubElement(first_row_tc, qn("w", "shd"), {qn("w", "val"): "clear", qn("w", "fill"): "EDEDED"})
    first_row_borders = ET.SubElement(first_row_tc, qn("w", "tcBorders"))
    ET.SubElement(
        first_row_borders,
        qn("w", "bottom"),
        {
            qn("w", "val"): "single",
            qn("w", "sz"): "8",
            qn("w", "space"): "0",
            qn("w", "color"): "606060",
        },
    )
    ET.SubElement(first_row_tc, qn("w", "vAlign"), {qn("w", "val"): "center"})


def polish_document(document_root: ET.Element) -> None:
    body = document_root.find(qn("w", "body"))
    if body is None:
        return

    body_children = list(body)
    merged_affiliation_paragraph: ET.Element | None = None
    for idx, child in enumerate(body_children):
        if child.tag != qn("w", "p"):
            continue
        text = paragraph_text(child)
        if not text:
            continue
        if text.startswith("Nikita V. Laptev") and "Viacheslav V. Danilov" in text:
            set_paragraph_style(child, "Author")
        elif text.startswith("¹ Siberian State Medical University"):
            merged_affiliation_paragraph = child
        elif text.startswith("Corresponding author:"):
            set_paragraph_style(child, "Correspondence")
        elif text == "Abstract":
            set_paragraph_style(child, "AbstractTitle")
        elif text.startswith("Accurate intraoperative delineation of the aortic root"):
            set_paragraph_style(child, "Abstract")
        elif text.startswith("Keywords:"):
            set_paragraph_style(child, "Keywords")
        elif text.startswith("Table ") and idx + 1 < len(body_children) and body_children[idx + 1].tag == qn("w", "tbl"):
            set_paragraph_style(child, "TableCaption")
            remove_direct_bold(child)

    if merged_affiliation_paragraph is not None:
        insert_at = list(body).index(merged_affiliation_paragraph)
        body.remove(merged_affiliation_paragraph)
        for offset, affiliation in enumerate(AFFILIATIONS):
            body.insert(insert_at + offset, make_text_paragraph(affiliation, "Affiliation"))

    sect_pr = body.find(qn("w", "sectPr"))
    if sect_pr is None:
        sect_pr = ET.SubElement(body, qn("w", "sectPr"))
    else:
        for node in list(sect_pr):
            sect_pr.remove(node)
    ET.SubElement(sect_pr, qn("w", "pgSz"), {qn("w", "w"): "11906", qn("w", "h"): "16838"})
    ET.SubElement(
        sect_pr,
        qn("w", "pgMar"),
        {
            qn("w", "top"): "1440",
            qn("w", "right"): "1440",
            qn("w", "bottom"): "1440",
            qn("w", "left"): "1440",
            qn("w", "header"): "708",
            qn("w", "footer"): "708",
            qn("w", "gutter"): "0",
        },
    )
    ET.SubElement(sect_pr, qn("w", "cols"), {qn("w", "space"): "708"})
    ET.SubElement(sect_pr, qn("w", "docGrid"), {qn("w", "linePitch"): "360"})


def polish_docx(docx_path: Path) -> None:
    with ZipFile(docx_path, "r") as src:
        files = {name: src.read(name) for name in src.namelist()}

    styles_root = ET.fromstring(files["word/styles.xml"])
    document_root = ET.fromstring(files["word/document.xml"])

    polish_styles(styles_root)
    polish_document(document_root)

    files["word/styles.xml"] = ET.tostring(styles_root, encoding="utf-8", xml_declaration=True)
    files["word/document.xml"] = ET.tostring(document_root, encoding="utf-8", xml_declaration=True)

    with tempfile.NamedTemporaryFile(
        prefix="cmig_docx_polished_",
        suffix=".docx",
        dir=docx_path.parent,
        delete=False,
    ) as tmp:
        temp_docx = Path(tmp.name)

    with ZipFile(temp_docx, "w", compression=ZIP_DEFLATED) as dst:
        for name, data in files.items():
            dst.writestr(name, data)

    temp_docx.replace(docx_path)


def build_markdown(tempdir: Path) -> Path:
    tex = TEX_PATH.read_text()
    tex = strip_comments(tex)

    title, _ = extract_brace_arg(tex, r"\title{")
    abstract = find_block(tex, r"\begin{abstract}", r"\end{abstract}").strip()
    keywords = find_block(tex, r"\begin{keyword}", r"\end{keyword}").strip()

    body = find_block(tex, r"\section{Introduction}", r"\bibliographystyle{elsarticle-harv}")
    body = r"\section{Introduction}" + body

    fig1_path = tempdir / "Figure_1_composite.png"
    fig3_path = tempdir / "Figure_3_composite.png"
    combine_images(
        [
            CMIG_DIR / "figures" / "Figure_1a.png",
            CMIG_DIR / "figures" / "Figure_1b.jpeg",
            CMIG_DIR / "figures" / "Figure_1c.png",
            CMIG_DIR / "figures" / "Figure_1d.jpeg",
        ],
        fig1_path,
        cols=2,
        labels=["(a)", "(b)", "(c)", "(d)"],
    )
    combine_images(
        [
            CMIG_DIR / "figures" / "Figure_3a.png",
            CMIG_DIR / "figures" / "Figure_3b.png",
        ],
        fig3_path,
        cols=2,
        labels=["(a)", "(b)"],
    )

    figure_replacements = [
        (
            re.compile(
                r"\\begin\{figure\}\[htbp\].*?\\label\{fig:annotation_examples\}\s*\\end\{figure\}",
                flags=re.S,
            ),
            figure_block_markdown(
                "Figure 1",
                fig1_path.name,
                "Representative fluoroscopic images (a, c) and corresponding expert annotations (b, d) illustrating the aortic root mask and four anatomical landmarks (AA1, AA2, STJ1, STJ2).",
            ),
        ),
        (
            re.compile(
                r"\\begin\{figure\}\[htbp\].*?\\label\{fig:bamnet_architecture\}\s*\\end\{figure\}",
                flags=re.S,
            ),
            figure_block_markdown(
                "Figure 2",
                "figures/Figure_2.png",
                "Overview of the proposed BoundaryAwareMANet (BAMNet) architecture for joint aortic root segmentation and anatomical landmark localization on fluoroscopic images. The network combines an EfficientNet-V2 encoder, a segmentation-oriented decoder, a dedicated landmark branch with Coordinate Attention, and an auxiliary boundary pathway used for boundary-guided supervision.",
            ),
        ),
        (
            re.compile(
                r"\\begin\{figure\}\[htbp\].*?\\label\{fig:bamnet_qualitative\}\s*\\end\{figure\}",
                flags=re.S,
            ),
            figure_block_markdown(
                "Figure 3",
                fig3_path.name,
                "Representative BAMNet predictions on intraoperative fluoroscopic frames. The yellow contour and semi-transparent green region denote the predicted aortic root mask. Yellow points correspond to the four predicted landmarks (AA1, AA2, STJ1, STJ2). The blue arrow indicates the estimated aortic axis, and the central geometric overlay illustrates the predicted implantation zone derived from the landmark configuration.",
            ),
        ),
        (
            re.compile(
                r"\\begin\{figure\}\[htbp\].*?\\label\{fig:ablation_summary\}\s*\\end\{figure\}",
                flags=re.S,
            ),
            figure_block_markdown(
                "Figure 4",
                "figures/Figure_4.png",
                "Summary of the BAMNet ablation study across the main architectural variants. The figure highlights how global spatial attention, coordinate-aware modulation, feature fusion, boundary-related supervision, and the soft-argmax configuration affect the balance between segmentation quality and landmark localization accuracy.",
            ),
        ),
    ]
    for pattern, replacement in figure_replacements:
        body = pattern.sub(replacement, body)

    body = re.sub(r"\\begin\{table\}\[htbp\].*?\\end\{table\}", table_env_to_markdown, body, flags=re.S)
    body = resolve_refs(body)
    body = escape_metric_at(body)
    body = convert_equations(body)
    body = convert_lists(body)
    body = re.sub(r"\\section\{([^{}]+)\}", r"\n\n# \1\n\n", body)
    body = re.sub(r"\\subsection\{([^{}]+)\}", r"\n\n## \1\n\n", body)
    body = re.sub(r"\\section\*\{([^{}]+)\}", r"\n\n# \1\n\n", body)
    body = re.sub(r"\\subsection\*\{([^{}]+)\}", r"\n\n## \1\n\n", body)
    body = convert_inline_latex(body)
    body = body.replace("`", "")
    body = normalize_markdown(body)

    author_block = "\n\n".join(
        [
            "Nikita V. Laptev¹*; Olga M. Gerget²; Julia K. Panteleeva³; Mikhail A. Chernyavsky³; Viacheslav V. Danilov⁴",
            *AFFILIATIONS,
            "*Corresponding author:* Nikita V. Laptev (nikitalaptev77@gmail.com); Olga M. Gerget (olgagerget@mail.ru); Viacheslav V. Danilov (viacheslav.v.danilov@gmail.com)",
        ]
    )

    markdown = "\n".join(
        [
            "---",
            f'title: "{latex_accents(title)}"',
            "numbersections: true",
            "reference-section-title: References",
            "---",
            "",
            author_block,
            "",
            "## Abstract",
            "",
            normalize_markdown(convert_inline_latex(abstract)),
            "",
            f"**Keywords:** {normalize_markdown(convert_inline_latex(keywords, keep_citations=False)).replace(r'\\sep', '; ').replace(r'\sep', '; ').replace(' ;', ';').strip()}",
            "",
            body.strip(),
            "",
        ]
    )

    md_path = tempdir / "manuscript_controlled.md"
    md_path.write_text(markdown)
    return md_path


def run_pandoc(md_path: Path, tempdir: Path) -> None:
    cmd = [
        "pandoc",
        str(md_path),
        "--from=markdown+raw_tex+tex_math_dollars+pipe_tables+table_captions+link_attributes+implicit_figures",
        "--to=docx",
        f"--resource-path={tempdir}:{CMIG_DIR}",
        f"--bibliography={BIB_PATH}",
        "--citeproc",
        "-o",
        str(OUTPUT_PATH),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cmig_docx_") as tmp:
        tempdir = Path(tmp)
        md_path = build_markdown(tempdir)
        run_pandoc(md_path, tempdir)
    polish_docx(OUTPUT_PATH)
    print(f"Generated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
