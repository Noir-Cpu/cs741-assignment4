# Citation verification

Policy: a source stays in the report only when its metadata (authors, title, venue, year, pages) matched CrossRef or the
publisher page AND every claim that the text attributes to the source is shown by the abstract, the publisher page or the course PDF.
`references.bib` carries the DOI of each entry in a comment. The items below are NOT cited in the text.

| Item | Why it is not used |
|---|---|
| Nadeau, C. and Bengio, Y. (2003), Inference for the generalization error, Machine Learning 52(3), 239-281, DOI 10.1023/A:1024068626366 | Metadata verified with CrossRef, but no abstract was reachable and the course PDFs do not mention the paper, so the claim "variance correction" could not be checked. The corrected resampled t-test is stated through its equation without a citation. |
| Aggarwal, C. C., Hinneburg, A. and Keim, D. A. (2001), On the surprising behavior of distance metrics in high dimensional space, DOI 10.1007/3-540-44503-X_27 | Metadata verified, abstract not reachable; the claim about the Manhattan distance was removed from the report. |
| Shepard, D. (1968), A two-dimensional interpolation function for irregularly-spaced data, DOI 10.1145/800186.810616 | The abstract does not show the inverse-distance weighting that the report states; the citation was removed. |
| Sokolova, M. and Lapalme, G. (2009), A systematic analysis of performance measures for classification tasks, DOI 10.1016/j.ipm.2009.03.002 | No abstract reachable; the citation was removed. |
| Wolpert, D. H. (1992), Stacked generalization, Neural Networks 5(2), 241-259, DOI 10.1016/S0893-6080(05)80023-1 | Metadata verified; the content was not read. The report cites the course slides for stacking instead. |
| Domingos, P. (2000), A unified bias-variance decomposition for zero-one and squared loss | Proceedings paper without a DOI; the decomposition follows the definitions of Breiman (1998), whose abstract was read. |
| Dietterich, T. G. (2000), An experimental comparison of three methods for constructing ensembles of decision trees: bagging, boosting, and randomization, Machine Learning 40(2), 139-157, DOI 10.1023/A:1007607513941 | Existence verified in CrossRef; publisher blocks automatic access, no abstract read. Not cited. |
| Bauer, E. and Kohavi, R. (1999), An empirical comparison of voting classification algorithms: bagging, boosting, and variants, Machine Learning 36(1-2), 105-139, DOI 10.1023/A:1007515423169 | Same reason. Not cited. |

## Kept with a limited claim (title level only; no abstract reachable)

| Item | Claim in the text |
|---|---|
| Breiman (1996), Bagging predictors, DOI 10.1007/BF00058655 | Only that Breiman introduced bagging predictors. |
| Wolpert and Macready (1999), DOI 10.1023/A:1007519102914 | Only that the paper proposes an efficient method to estimate the generalisation error of bagging (the title). |
| Kuncheva and Whitaker (2003), DOI 10.1023/A:1022859003006 | Only that the paper examines measures of diversity and their relationship with ensemble accuracy (the title). |
| Freund and Schapire (1997), DOI 10.1006/jcss.1997.1504 | Only the description of their algorithm that Breiman (1998) gives in his abstract, which is cited with it. |
| Quinlan (1986), DOI 10.1007/BF00116251 | Only that Quinlan described the induction of decision trees (the title). |
| Hastie, Rosset, Zhu and Zou (2009), Multi-class AdaBoost | Author order follows CrossRef; the scikit-learn documentation lists Zhu, Zou, Rosset, Hastie. Check on the publisher page before submission. |
| Breiman (1998), Annals of Statistics 26(3) and Friedman (2001), Annals of Statistics 29(5) | CrossRef holds no page data; pages 801-849 and 1189-1232 come from the Semantic Scholar record. |

## Round 2 additions

The entries holm1979, wolpert1992, dietterich2000, geurts2006, chen2016, ting2002 and chawla2002 were added in round 2.
Each one is cited only for what `report/CITATIONS_EVIDENCE.md` records as
established by CrossRef, the publisher page or the paper. The Iman and Davenport variant and the Nemenyi critical difference
are cited through Rainio et al. (course PDF) and Demsar (JMLR page). The Breiman (1998) pages are now in CrossRef (801-849).
`scripts/verify_bib.py` reports CHECK for the JMLR and IJCAI papers (no DOI; verified on the publisher pages), for single-field
author names in CrossRef (Ho, Ting) and for the truncated NumPy and SciPy author lists.

## Final audit (17 Oct)
All 34 cited keys checked with scripts/verify_bib.py. Remaining CHECK cases verified directly:
- **bergstra2012**: JMLR publisher page (jmlr.org/papers/v13/bergstra12a.html) confirms title, authors J. Bergstra and Y. Bengio, vol. 13, pp. 281-305 (JMLR has no DOI).
- **he2009**: CrossRef doi:10.1109/tkde.2008.239 confirms title, authors He and Garcia, vol. 21, pp. 1263-1284 (CHECK only because CrossRef stores "Haibo He" in one field).
- **liu2009**: CrossRef doi:10.1109/tsmcb.2008.2007853 confirms title, authors Liu, Wu and Zhou, vol. 39, pp. 539-550 (single-field names in CrossRef).
- harris2020a, holm1979, lam2015a, moustafa2015a, ting2002, virtanen2020a, chen2016, ho1998: verified earlier (see above and CITATIONS_EXTRA_EVIDENCE.md).
