# CS741 Assignment 4

Source code for the CS741 Assignment 4 report on the UNSW-NB15 attack category prediction task, reusing the data set supplied for Assignment 2.

## Layout

```
code/       experiment scripts and the from-scratch algorithm implementations
results/    every number reported in the paper, written by the scripts as CSV or JSON
figures/    every figure in the paper, written by the scripts as PDF
report/     LaTeX source of the report (IEEE conference format, two columns, 10 pt)
scripts/    lint_report.py (writing-rule checks) and check_pdf.sh (submission PDF gate)
data/       place networkTraffic.csv here; the file is not tracked
```

## Data

The course-supplied `networkTraffic.csv` is not redistributed. Copy it to `data/networkTraffic.csv` before running any script. Scripts read paths relative to the repository root.



