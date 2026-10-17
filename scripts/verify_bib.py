"""Match each cited BibTeX entry against CrossRef (title, first author surname, year).

Usage: python3 scripts/verify_bib.py a.bib,b.bib key1,key2,...
OK means the CrossRef record agrees on title, first author and year. CHECK needs a manual look
(JMLR and IJCAI papers have no DOI; verify those on the publisher page)."""
import json, re, sys, time, urllib.parse, urllib.request
from difflib import SequenceMatcher
from pathlib import Path

bibs, keys = sys.argv[1].split(","), set(sys.argv[2].split(","))


def parse(text):
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.groups()
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body):
            fields[f.group(1).lower()] = re.sub(r"[{}\\]", "", f.group(2)).strip()
        fields["_kind"] = kind
        out[key.strip()] = fields
    return out


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def crossref(title, author, year):
    q = urllib.parse.urlencode({"query.bibliographic": f"{title} {author} {year}", "rows": 3,
                                "select": "DOI,title,author,issued,container-title,volume,page,type"})
    req = urllib.request.Request("https://api.crossref.org/works?" + q,
                                 headers={"User-Agent": "citation-check (mailto:none@example.org)"})
    return json.load(urllib.request.urlopen(req, timeout=30))["message"]["items"]


entries = {}
for b in bibs:
    entries.update(parse(Path(b).read_text()))
rows = []
for key in sorted(keys):
    e = entries.get(key)
    if e is None:
        rows.append((key, "MISSING-IN-BIB", "", "", ""))
        continue
    title, year = e.get("title", ""), e.get("year", "")
    first = re.split(r"\s+and\s+", e.get("author", ""))[0]
    surname = first.split(",")[0].strip() if "," in first else first.split()[-1] if first else ""
    try:
        items = crossref(title, surname, year)
    except Exception as ex:
        rows.append((key, "NETWORK-ERROR", str(ex)[:60], "", ""))
        continue
    best = None
    for it in items:
        t = (it.get("title") or [""])[0]
        sim = SequenceMatcher(None, norm(title), norm(t)).ratio()
        au = norm((it.get("author") or [{}])[0].get("family", ""))
        yr = str((it.get("issued", {}).get("date-parts") or [[None]])[0][0])
        score = (sim, au == norm(surname), yr == year)
        if best is None or score > best[0]:
            best = (score, it, t, au, yr)
    if best is None:
        rows.append((key, "NO-HIT", title[:50], "", ""))
        continue
    (sim, au_ok, yr_ok), it, t, au, yr = best
    status = "OK" if sim > 0.9 and au_ok and yr_ok else "CHECK"
    venue = (it.get("container-title") or [""])[0][:40]
    rows.append((key, status, f"sim={sim:.2f} author={'y' if au_ok else 'n'} year={'y' if yr_ok else 'n'}",
                 f"{it['DOI']} | {venue} | {yr}", f"bib: {title[:55]} / {e.get('journal', e.get('booktitle', e.get('publisher', '')))[:40]}"))
    time.sleep(0.3)
for r in rows:
    print(" | ".join(r))
