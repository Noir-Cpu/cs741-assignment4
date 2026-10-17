# code/common: common library

From-scratch implementations (numpy and Numba) of the data handling, the
evaluation protocol, the baselines and the classification tree that the ensembles
build on. Scikit-learn appears only in `tests/` as a correctness oracle.

All scripts import the package like this (paths derive from `__file__`, no
absolute paths):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the code/ directory
from common import data, preprocess, protocol, metrics, stats, knn, tree, plotstyle
```

Set the thread counts before numpy is imported (each experiment uses at most
10 cores):

```python
import os
for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "10")
```

Run the tests with `.venv/bin/python -m pytest tests -q`.

## protocol.py: master seed, random streams, partitions

Master seed `MASTER_SEED = 1072417991`. Every stream is derived with
`numpy.random.SeedSequence(entropy=MASTER_SEED, spawn_key=sha256(purpose))`.
Never write a literal seed; name the purpose.

| Function | Purpose |
|---|---|
| `rng(purpose)` | independent `numpy.random.Generator` |
| `seed_int(purpose)` | 32-bit integer seed for Numba kernels and tree seeds |
| `run_seeds(purpose, n)` | seeds for `n` independent runs of a stochastic model; run `i` uses `"<purpose>/run<i>"`, so adding runs never changes earlier seeds |
| `stratified_holdout(y)` | `(dev_idx, gen_idx)`: G is a stratified 20 percent hold-out (purpose `generalisation-split`) |
| `tuning_partitions(y_dev)` | 5 stratified folds inside D (purpose `tuning-cv`) |
| `comparison_partitions(y_dev)` | 15 partitions: 5-fold CV repeated 3 times (purpose `comparison-cv`) |
| `stratified_kfold(y, n_folds, purpose, repeat)`, `repeated_stratified_kfold(...)` | generic versions for extra experiments |
| `stratified_subsample(y, n, purpose, min_per_class)` | class-proportional subsample (sorted indices) for tuning on a subset |
| `index_digest(idx)` | short hash to verify that a split did not change |

A `Partition` has `.repeat`, `.fold`, `.train`, `.test` (index arrays into the array
that was split, here D).

```python
table = data.load_table()
dev_idx, gen_idx = protocol.stratified_holdout(table.y)
dev, gen = table.take(dev_idx), table.take(gen_idx)
for part in protocol.comparison_partitions(dev.y):          # 15 paired partitions
    train, test = dev.take(part.train), dev.take(part.test)
```

Use `rng("<experiment>/<stream>")`-style purposes for new streams.

## data.py: loading

`load_table()` reads `data/networkTraffic.csv` into a `Table` (numpy only):
`numeric` (n, 39) float64 with `numeric_names`, `nominal` (n, 3) integer codes with
`nominal_names = ("proto", "service", "state")` and `vocab`, `y` (0 to 9), `ids`.
The identifier is not a feature. The missing symbol `?` (service only) is loaded as
the category `none`. `Table.take(idx)` selects rows, `.column(name)`,
`.nominal_values(name)`. Also `CLASS_NAMES`, `N_CLASSES`, `class_counts(y)`,
`REPO_ROOT`, `RESULTS_COMMON`.

## preprocess.py: fit on training rows only

```python
pre = preprocess.Preprocessor.for_knn()      # drops sloss/dloss/ct_ftp_cmd, log1p on 25 columns,
                                             # standardise, rare-proto grouping, one-hot (74 columns)
pre = preprocess.Preprocessor.for_tree()     # raw numeric, rare-proto grouping, one-hot (79 columns)
x_train = pre.fit_transform(train); x_test = pre.transform(test)    # float64 (n, d)
pre.feature_names_, pre.feature_kinds_       # "numeric" or "onehot:<column>" per output column
```

Flags: `drop_redundant`, `log_transform`, `standardise`, `group_rare`,
`rare_threshold` (default 150). Statistics (means, standard
deviations, category counts) come from the rows passed to `fit` only.

## knn.py: exact k-nearest neighbours

```python
model = knn.KNNClassifier(n_neighbors=5, metric="manhattan", weights="distance",
                          class_weight=None, n_classes=10).fit(x_train, y_train)
pred = model.predict(x_test)                 # also predict_proba, decision_scores
dist, idx = model.kneighbors(x_test, k=51)   # sorted, float32 distances and int32 indices
```

* `metric`: `"euclidean"` or `"manhattan"`; `weights`: `"uniform"` or `"distance"`
  (1 / max(d, 1e-12)); `class_weight`: `None`, `"balanced"` or a float power in
  (0, 1] applied to n / (m n_j).
* Tune cheaply: compute neighbours once with the largest k, then
  `KNNClassifier.predict_from_neighbours(idx, dist, y_train, k, weights, class_weight_vec, n_classes)`
  for every candidate (`knn.class_weight_vector(y, n_classes, power)` builds the vector).
* Memory: training matrix (float32) plus (n_query, k) arrays; the distance matrix is
  never formed. Speed: about 14 s for 41,000 queries against 165,000 training rows
  with k = 51 on 20 threads.
* Ties: lower training index for equal distances, lower class label for equal votes.

## tree.py: classification tree for use inside ensembles

```python
binner = tree.BinMapper(max_bins=255).fit(x_train)        # once per training set
xb_t = binner.transform(x_train)                          # feature-major uint8/uint16 (p, n)
model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=25, min_samples_leaf=1,
            max_features="sqrt", class_weight="balanced", seed=protocol.seed_int("ensemble/tree0"))
model.fit(x_train, y_train, sample_weight=bootstrap_counts, n_classes=10,
          binner=binner, Xb_t=xb_t)                       # binner/Xb_t optional
proba = model.predict_proba(x_test)    # normalised weighted leaf totals
pred = model.predict(x_test)           # argmax of the weighted leaf totals
leaf = model.apply(x_test)             # leaf index per row
```

* `criterion` gini or entropy; `max_depth` (None unlimited); `min_samples_leaf`
  (rows, not weight); `min_samples_split`; `max_features` (int, fraction, `"sqrt"`,
  `"log2"`, None); `class_weight` (None, `"balanced"`, dict, array); per-row
  `sample_weight` (rows with weight 0 are left out, so pass bootstrap multiplicities
  to grow a tree on a bootstrap sample and use the zero-weight rows as the
  out-of-bag set).
* Attributes after fit: `feature_`, `threshold_`, `children_left_`,
  `children_right_`, `value_` (weighted class totals per node), `n_node_samples_`,
  `n_nodes_`, `n_leaves_`, `depth_`, `feature_importances_`.
* Thread safety: the fit and the serial `predict` release the GIL, so a
  `ThreadPoolExecutor` fits many trees in parallel, each on one thread. Share one
  `BinMapper` and one binned matrix across all trees. Use `predict(..., parallel=True)`
  only outside such a pool.
* `max_bins`: 65,535 (default) is close to an exact split search (a 255-bin tree lost
  about 0.01 macro-F1 on one fold); 255 makes data uint8 and fits roughly 3 times faster.
  A baseline fit on 165,000 rows takes 2 to 3 seconds on one thread.
* Deterministic: the result depends only on data, parameters and `seed`. With all
  features tried at every split the seed has no effect. Default seed: purpose
  `tree-default`.
* Differences from scikit-learn: thresholds are global bin boundaries, features with
  more than `max_bins` distinct values are quantised, and the random feature order
  differs. Predictions agree with scikit-learn on 97 to 100 percent of rows in
  `tests/common/test_common_tree.py`.

## metrics.py

`confusion_matrix(y_true, y_pred, n_classes)` (rows true, columns predicted),
`scores_from_confusion(cm)` and `score_all(y_true, y_pred, n_classes)` return a dict with
`accuracy`, `macro_f1`, `weighted_f1`, `balanced_accuracy`. `per_class(cm)` returns
precision, recall, F1, support; `per_class_table(cm)` returns rows for a CSV;
`normalise_rows(cm)`. Convention: zero division scores 0; macro averages run over all
`n_classes` classes; balanced accuracy averages over classes with support.

```python
cm = metrics.confusion_matrix(y_test, pred, 10)
scores = metrics.scores_from_confusion(cm)
```

## stats.py

* `paired_wilcoxon(a, b)`: statistic, p-value, mean and median difference, paired
  Cohen's d, rank-biserial correlation.
* `corrected_resampled_ttest(a, b, test_train_ratio=0.25)`: Nadeau and Bengio
  variance correction for overlapping training sets.
* `friedman_test(scores)` (rows = partitions, columns = models; at least 3 models):
  chi-square and Iman-Davenport statistics with p-values and average ranks.
* `nemenyi_posthoc(scores)`: critical difference and pairwise p-values.
* `holm_wilcoxon_posthoc(scores, names)`, `holm_adjust(p)`, `summarise(x)`.

```python
res = stats.paired_wilcoxon(scores_tree["macro_f1"], scores_knn["macro_f1"])
```

## plotstyle.py

`plotstyle.apply()` sets Nimbus Roman (or TeX Gyre Termes), 10 pt everywhere, STIX
mathematics, embedded fonts, no figure titles and a colour-blind-safe palette.
`plotstyle.figure("single"|"double", height=...)` creates 3.5 in or 7.16 in wide
figures, `plotstyle.save(fig, path)` writes a vector PDF, `plotstyle.colour(name)`
returns a stable colour per model name.

## Scripts that produce results/common

| Script | Output |
|---|---|
| `run_data_quality.py` | `results/common/data_quality/` (class distribution, missing values, cardinality, duplicates, correlation, scale, skewness) |
| `run_baselines.py` | tuning grids and best parameters, 15 paired scores, per-class scores, confusion matrices, generalisation scores, statistics, runtimes |
| `run_preprocessing_ablation.py` | `preprocessing_ablation.csv` |
| `make_report_numbers.py` | `report/common/numbers.tex` (data-set and baseline numbers used by the report) |

Run from the repository root with `.venv/bin/python code/common/<script>.py`; long runs
under `nohup ... > logs/<name>.log 2>&1 &`.

## Report numbers (report/common)

`make_report_numbers.py` writes `report/common/numbers.tex`, which defines the data-set and baseline numbers of
the report as macros generated from `results/common`; `report/preamble.tex` loads it. Regenerate it after changing
anything in `results/common`.

## Grouped protocol (grouped.py): leakage-aware, additional protocol

The primary protocol above stays the reference. In it, 44 percent of the test rows of a fold have an
identical feature vector (a twin) in the training part. `grouped.py` adds a second protocol in which
identical vectors always stay together, so no test row has a twin in training. Report both and state
whether the model ranking changes. Never replace the primary protocol.

Groups: all 42 descriptive columns (39 numeric as float64, 3 nominal codes; id and target excluded) are
written as float64, hashed per row with BLAKE2b (128 bit) and rows with equal digests form a group.
Groups with conflicting labels stay whole. Partitions use the same `protocol.Partition` objects, so
existing loops work unchanged. Seeds come from `protocol.rng` with the purposes `grouped-generalisation-split`,
`grouped-tuning-cv/repeat0` and `grouped-comparison-cv/repeat<r>`.

| Function | Purpose |
|---|---|
| `feature_groups(table) -> int64 (n,)` | group number per row (first-occurrence order); about 2.4 s for the full table |
| `grouped_split(table) -> GroupedSplit` | `.groups`, `.dev_idx`, `.gen_idx`, `.groups_dev`, `.groups_gen` in one call |
| `grouped_holdout(table, purpose=..., groups=None) -> (dev_idx, gen_idx)` | D_g and G_g (sorted indices into the table); class counts of G_g match the shared G within a few rows, group structure matches D_g |
| `grouped_tuning_partitions(y_dev, groups_dev) -> list[Partition]` | 5 folds inside D_g |
| `grouped_comparison_partitions(y_dev, groups_dev) -> list[Partition]` | 15 partitions (5 folds x 3 repeats, repeat-major) |
| `grouped_kfold(y, groups, n_folds, purpose, repeat)` | generic version for extra experiments |
| `twin_share(groups, partition)`, `twin_share_same_label(groups, y, partition)`, `groups_cross(groups, partition)` | leakage measurements |
| `balance_table(y, partitions)`, `class_balance_report(y, partitions)` | achieved class balance of the test folds |
| `group_summary(groups, y)` | repeated and conflicting groups of a set of rows |

Algorithm: groups are indivisible items; they are placed greedily (rarest class first, large groups first,
random order inside a class and size class) into the bin with the smallest increase of the squared relative
deviation from the targets (class counts, row count, group-size structure), followed by a local search that moves
single groups while the cost falls. The achieved balance is within a few rows of ideal for every class
(class 7: 27 or 28 test rows per fold).

```python
from common import data, protocol, grouped
table = data.load_table()
split = grouped.grouped_split(table)                     # D_g, G_g and the group numbers
dev, gen = table.take(split.dev_idx), table.take(split.gen_idx)
for part in grouped.grouped_comparison_partitions(dev.y, split.groups_dev):    # 15 paired partitions
    train, test = dev.take(part.train), dev.take(part.test)   # no group occurs in both
```

Models are fitted and evaluated exactly as in the primary protocol. Results of the baselines:
`results/common/grouped_*` (`run_grouped_baselines.py`; `grouped_ranking_comparison.csv` is the side-by-side table,
`grouped_conclusion.json` the verdict, `grouped_retuning.json` the retuning check, `grouped_twin_share.csv` the twin shares).
`make_grouped_numbers.py` writes the grouped-protocol macros to `report/common/numbers_grouped.tex`.
Scripts must keep the thread counts set before numpy is imported, as above.
