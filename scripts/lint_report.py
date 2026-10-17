#!/usr/bin/env python3
"""Heuristic check of a LaTeX report against the course writing rules (WRx).

Usage:
    python3 scripts/lint_report.py report/main.tex --bib report/references.bib \
        --student 22677267 --forbid "Surname" --forbid "firstname" --code code

The checks are regular expressions, so they flag candidates for review, not
verdicts. Rules marked ERROR are mechanical and should always be fixed.
The exit status is 1 when any ERROR is found.
"""
import argparse
import re
import sys
from pathlib import Path


def blank(match):
    """Replace matched text with spaces, keeping newlines so line numbers survive."""
    return re.sub(r"[^\n]", " ", match.group(0))


def sanitise(src):
    """Remove comments, math, and command arguments that are not prose."""
    src = re.sub(r"(?<!\\)%.*", blank, src)
    src = re.sub(r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}.*?\\end\{\1\*?\}", blank, src, flags=re.S)
    src = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]", blank, src, flags=re.S)
    src = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", blank, src, flags=re.S)
    src = re.sub(r"\\(cite[a-z]*|ref|eqref|autoref|cref|label|url|input|include|bibliography|bibliographystyle|glsadd)\*?(\[[^\]]*\])?\{[^}]*\}", blank, src)
    src = re.sub(r"\\(usepackage|documentclass|includegraphics)(\[[^\]]*\])?\{[^}]*\}", blank, src)
    src = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", blank, src)
    return src


def word(pattern):
    return r"(?<![\\\w])(?:" + pattern + r")(?!\w)"


# (rule, level, message, regex, flags)
TEXT_RULES = [
    ("WR3", "WARN", "first person or 'one'", word(r"we|We|our|Our|ours|us|my|I|[Oo]ne (?:can|may|should|must|could|would)"), 0),
    ("WR5", "WARN", "ambiguous pronoun (it, its, this, these, them, they, which)", word(r"it|its|this|these|them|they|which"), re.I),
    ("WR6", "WARN", "uncertain term (some, certain)", word(r"some|certain"), re.I),
    ("WR7", "WARN", "'may be' or 'can be'", word(r"may be|can be|might be"), re.I),
    ("WR8", "WARN", "'etc'", word(r"etc"), re.I),
    ("WR9", "WARN", "footnote (allowed only for URLs to code or data)", r"\\footnote(?:text)?\b", 0),
    ("WR13", "ERROR", "contraction", r"(?<=\w)(?:n't|'re|'ve|'ll|'d|'m)(?!\w)|(?<![\\\w])[Ii]t's(?!\w)", 0),
    ("WR14", "WARN", "apostrophe used for position", r"(?<=[A-Za-z])'s(?!\w)|(?<=[a-z]s)'(?=\s)", 0),
    ("WR17", "WARN", "check conjunction use (since, because)", word(r"since|because"), re.I),
    ("WR18", "WARN", "single digit written as a numeral", r"(?<![\w.\-{\\])[1-9](?![\w.%\-}])(?!\.\d)", 0),
    ("WR19c", "WARN", "sentence starts with a number or symbol", r"(?:[.!?]\s+)(?:\d|\$)", 0),
    ("WR19f", "WARN", "sentence opens with 'In this section/report'", word(r"In this (?:section|chapter|report|paper|study)"), 0),
    ("WR19g", "ERROR", "sentence starts with And or But", r"(?:^|[.!?]\s+)(?:And|But)\b", re.M),
    ("WR30", "WARN", "'et al' not in italics", r"(?<!\\textit\{)(?<!\\emph\{)(?<!\{)\bet al\b", 0),
    ("WR22c", "WARN", "acronym in heading or caption", r"\\(?:sub)*section\*?\{[^}]*\b[a-z]*[A-Z]{2,}[a-z]*\b[^}]*\}|\\caption\{[^}]*\b[a-z]*[A-Z]{2,}[a-z]*\b[^}]*\}", 0),
    ("WR28a", "ERROR", "heading is a question", r"\\(?:sub)*section\*?\{[^}]*\?[^}]*\}", 0),
    ("MARKER", "WARN", "competitive or anthropomorphic wording about results", word(r"wins?|beats?|beaten|outperform\w*|jumps?|jumped|crosses|crossed"), re.I),
    ("PLACEMENT", "ERROR", "forced float placement (use t or b, never h or H)", r"\\begin\{(?:figure|table)\*?\}\[[^\]]*[hH][^\]]*\]", 0),
    ("PREAMBLE", "ERROR", "hyperlinks must be deactivated: do not load hyperref", r"\\usepackage(?:\[[^\]]*\])?\{[^}]*hyperref[^}]*\}|\\href\b", 0),
    ("IDENTITY", "ERROR", "email address in report", r"[\w.+-]+@[\w-]+\.[\w.-]+", 0),
]


# Structural rules that must see commands the sanitiser blanks out.
RAW_RULES = {"PLACEMENT", "PREAMBLE", "IDENTITY"}


def check_floats(raw, report):
    """WR25a and WR26a: figure captions below, table captions above."""
    for m in re.finditer(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", raw, flags=re.S):
        body, kind = m.group(0), m.group(1)
        cap = [c.start() for c in re.finditer(r"\\caption\b", body)]
        if not cap:
            continue
        line = raw.count("\n", 0, m.start()) + 1
        if kind == "figure":
            last_graphic = max([g.start() for g in re.finditer(r"\\includegraphics|\\end\{subfigure\}", body)] or [-1])
            if cap[-1] < last_graphic:
                report("WR25a", "ERROR", line, "figure caption must be below the figure")
        else:
            first_body = min([t.start() for t in re.finditer(r"\\begin\{tabular", body)] or [10**9])
            if cap[0] > first_body:
                report("WR26a", "ERROR", line, "table caption must be above the table")


def check_sections(raw, report):
    """WR28c: a main section must not have exactly one subsection."""
    parts = re.split(r"(?m)^(?=\\section\b)", raw)
    for part in parts:
        if part.startswith("\\section"):
            n = len(re.findall(r"\\subsection\b", part))
            if n == 1:
                line = raw.count("\n", 0, raw.find(part)) + 1
                report("WR28c", "ERROR", line, "main section has exactly one subsection")


def check_title(raw, student, report):
    """The title section must carry the student number and no identity."""
    head = raw.split("\\begin{abstract}")[0]
    if student and student not in head:
        report("TITLE", "ERROR", 1, f"student number {student} not found before the abstract")


def check_bib(path, report):
    text = path.read_text(errors="replace")
    for m in re.finditer(r"(?i)arxiv|preprint|corr\b|researchsquare|ssrn|biorxiv|blog|medium\.com", text):
        line = text.count("\n", 0, m.start()) + 1
        report("WR23d", "ERROR", line, f"preprint or blog source in bibliography: '{m.group(0)}'", path)
    for m in re.finditer(r"(?is)\bauthor\s*=\s*[{\"](.*?)[}\"]\s*,?\s*\n", text):
        for name in re.split(r"\s+and\s+", m.group(1)):
            given = name.split(",", 1)[1] if "," in name else ""
            if re.search(r"[a-z]{2,}", given):
                line = text.count("\n", 0, m.start()) + 1
                report("WR23c", "ERROR", line, f"author given name not reduced to initials: '{name.strip()}'", path)
    for m in re.finditer(r"(?is)@inproceedings\{.*?\n\}", text):
        if not re.search(r"(?i)booktitle\s*=\s*[{\"]\s*Proceedings of", m.group(0)):
            line = text.count("\n", 0, m.start()) + 1
            report("WR23i", "WARN", line, "booktitle should read 'Proceedings of the ...'", path)


def check_code(root, report):
    seed = re.compile(r"\w*(?:random_state|seed|default_rng|RandomState)\w*\s*[=(]\s*(?:42|0|1|7|123|1234|12345|2024|2025|2026)\b", re.I)
    for path in sorted(Path(root).rglob("*.py")):
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if seed.search(line):
                report("SEED", "ERROR", i, f"generic random seed: {line.strip()}", path)
            if re.match(r"\s*(?:from|import)\s+sklearn", line):
                report("SKLEARN", "WARN", i, f"scikit-learn import, confirm it is not the algorithm under study: {line.strip()}", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tex")
    ap.add_argument("--bib")
    ap.add_argument("--code")
    ap.add_argument("--student")
    ap.add_argument("--forbid", action="append", default=[], help="case-insensitive regex for identifying text")
    args = ap.parse_args()

    findings = []

    def report(rule, level, line, msg, path=None):
        findings.append((rule, level, line, msg, path or Path(args.tex)))

    raw = Path(args.tex).read_text(errors="replace")
    clean = sanitise(raw)
    lines = clean.splitlines()
    uncommented = re.sub(r"(?<!\\)%.*", blank, raw)
    for rule, level, msg, pattern, flags in TEXT_RULES:
        source = uncommented if rule in RAW_RULES else clean
        for m in re.finditer(pattern, source, flags):
            line = source.count("\n", 0, m.start()) + 1
            snippet = source.splitlines()[line - 1].strip()[:90]
            report(rule, level, line, f"{msg}: {snippet}")
    for pattern in args.forbid:
        for m in re.finditer(pattern, raw, re.I):
            report("IDENTITY", "ERROR", raw.count("\n", 0, m.start()) + 1, f"identifying text matches '{pattern}'")
    check_floats(raw, report)
    check_sections(raw, report)
    check_title(raw, args.student, report)
    if args.bib:
        check_bib(Path(args.bib), report)
    if args.code:
        check_code(args.code, report)

    counts = {}
    for rule, level, line, msg, path in sorted(findings, key=lambda f: (f[0], str(f[4]), f[2])):
        counts[(rule, level)] = counts.get((rule, level), 0) + 1
        print(f"{level:5} {rule:9} {path}:{line}: {msg}")
    print("\nSummary")
    for (rule, level), n in sorted(counts.items()):
        print(f"  {level:5} {rule:9} {n}")
    errors = sum(n for (_, level), n in counts.items() if level == "ERROR")
    print(f"\n{errors} ERROR finding(s), {sum(counts.values()) - errors} WARN finding(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
