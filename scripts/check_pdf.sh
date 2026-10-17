#!/usr/bin/env bash
# Final gate for the submission PDF: no link annotations, student number present,
# no identifying text on the first page, page count and font report.
# Usage: scripts/check_pdf.sh report/main.pdf 22677267 "Surname|Firstname"
set -u
pdf="$1"; student="$2"; forbid="${3:-}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
status=0

mutool clean -d -g "$pdf" "$tmp/plain.pdf" 2>/dev/null
links=$(grep -a -c '/Subtype */Link' "$tmp/plain.pdf")
if [ "$links" -eq 0 ]; then echo "ok    no link annotations"; else echo "FAIL  $links link annotations (remove hyperref)"; status=1; fi

pdftotext -l 1 "$pdf" "$tmp/p1.txt"
if grep -q "$student" "$tmp/p1.txt"; then echo "ok    student number on page 1"; else echo "FAIL  student number not on page 1"; status=1; fi

if [ -n "$forbid" ] && pdftotext "$pdf" - | grep -E -i -q "$forbid"; then
  echo "FAIL  identifying text found"; status=1
elif [ -n "$forbid" ]; then echo "ok    no identifying text"; fi

if pdftotext "$pdf" - | grep -q 'github.com/Noir-Cpu/cs741-assignment4'; then echo "ok    repository link present"; else echo "FAIL  repository link missing (brief requires it)"; status=1; fi

if pdftotext "$pdf" - | grep -q '[?][?]'; then echo "FAIL  unresolved reference (??)"; status=1; else echo "ok    no unresolved references"; fi

if pdffonts "$pdf" | grep -E -i -q 'DejaVu|Arial|Helvetica|Liberation|Calibri|Verdana'; then
  echo "FAIL  figure font differs from the IEEEtran body font (match figure fonts to the main text, WR25b)"; status=1
else echo "ok    figure fonts match the body font"; fi

echo "info  $(pdfinfo "$pdf" | grep -E '^Pages')"
echo "info  fonts:"; pdffonts "$pdf" | awk 'NR>2 {print "        " $1}' | sort -u
exit $status
