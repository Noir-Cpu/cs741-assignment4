"""Data-quality analysis of the development set D (descriptive statistics only).

Reads data/networkTraffic.csv, applies the shared hold-out split, and computes
the data-quality findings on D only, so that the generalisation set G is not
inspected. Writes CSV and JSON files to results/common/data_quality/.

Usage (from the repository root):
    .venv/bin/python code/common/run_data_quality.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data, preprocess, protocol  # noqa: E402


def skewness(x: np.ndarray) -> float:
    """Fisher skewness coefficient (population moments)."""
    centred = x - x.mean()
    s2 = (centred ** 2).mean()
    return float((centred ** 3).mean() / s2 ** 1.5) if s2 > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(data.RESULTS_COMMON / "data_quality"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    full = data.load_table()
    dev_index, gen_index = protocol.stratified_holdout(full.y)
    tab = full.take(dev_index)
    n = tab.n
    report: dict = {"n_rows_file": full.n, "n_rows_development": n,
                    "n_numeric_features": len(tab.numeric_names),
                    "n_nominal_features": len(tab.nominal_names)}

    # class distribution
    counts_all = data.class_counts(full.y)
    counts_dev = data.class_counts(tab.y)
    pd.DataFrame({"class": range(data.N_CLASSES), "name": data.CLASS_NAMES,
                  "count_file": counts_all, "count_development": counts_dev,
                  "share_file": counts_all / full.n}).to_csv(out / "class_distribution.csv", index=False)
    report["imbalance_ratio_file"] = float(counts_all.max() / counts_all.min())

    # missing values: the symbol "?" was loaded as the category "none"
    j_service = tab.nominal_names.index("service")
    none_code = int(np.flatnonzero(tab.vocab[j_service] == data.MISSING_CATEGORY)[0])
    missing = tab.nominal[:, j_service] == none_code
    report["service_missing_count_development"] = int(missing.sum())
    report["service_missing_share_development"] = float(missing.mean())
    report["service_missing_share_file"] = float(
        (full.nominal[:, j_service] == none_code).mean())
    report["service_missing_count_file"] = int((full.nominal[:, j_service] == none_code).sum())
    for name in ("proto", "state"):
        j = tab.nominal_names.index(name)
        rows = []
        for code, label in enumerate(tab.vocab[j]):
            sel = tab.nominal[:, j] == code
            if sel.sum() == 0:
                continue
            rows.append({name: label, "count": int(sel.sum()),
                         "missing_service_share": float(missing[sel].mean())})
        crosstab = pd.DataFrame(rows).sort_values("count", ascending=False)
        crosstab.to_csv(out / f"service_missing_by_{name}.csv", index=False)
        report[f"service_missing_share_by_{name}_min"] = float(crosstab["missing_service_share"].min())
        report[f"service_missing_share_by_{name}_max"] = float(crosstab["missing_service_share"].max())
        report[f"service_missing_share_by_{name}_n_categories"] = int(len(crosstab))
    other_missing = int(np.isnan(tab.numeric).sum())
    report["missing_numeric_values"] = other_missing

    # cardinality of nominal features, rare protocol values
    card = {}
    for j, name in enumerate(tab.nominal_names):
        card[name] = int((np.bincount(tab.nominal[:, j], minlength=len(tab.vocab[j])) > 0).sum())
    report["nominal_cardinality_development"] = card
    j_proto = tab.nominal_names.index("proto")
    proto_counts = np.bincount(tab.nominal[:, j_proto], minlength=len(tab.vocab[j_proto]))
    proto_table = pd.DataFrame({"proto": tab.vocab[j_proto], "count": proto_counts})
    proto_table = proto_table[proto_table["count"] > 0].sort_values("count", ascending=False)
    proto_table.to_csv(out / "proto_counts.csv", index=False)
    rare = proto_table["count"] < preprocess.RARE_THRESHOLD
    report["proto_rare_threshold"] = preprocess.RARE_THRESHOLD
    report["proto_n_rare_categories"] = int(rare.sum())
    report["proto_rows_in_rare_categories"] = int(proto_table.loc[rare, "count"].sum())
    report["proto_categories_after_grouping"] = int((~rare).sum() + (1 if rare.any() else 0))
    report["proto_top3_share"] = float(proto_table["count"].iloc[:3].sum() / n)

    # binary feature irregularities
    ftp = tab.column("is_ftp_login")
    cmd = tab.column("ct_ftp_cmd")
    values, value_counts = np.unique(ftp, return_counts=True)
    report["is_ftp_login_values"] = {str(int(v)): int(c) for v, c in zip(values, value_counts)}
    out_of_range = ~np.isin(ftp, (0, 1))
    report["is_ftp_login_out_of_range_rows"] = int(out_of_range.sum())
    report["is_ftp_login_equals_ct_ftp_cmd_share"] = float((ftp == cmd).mean())
    report["is_ftp_login_equals_ct_ftp_cmd_on_out_of_range"] = bool(
        out_of_range.sum() > 0 and (ftp[out_of_range] == cmd[out_of_range]).all())
    sm = tab.column("is_sm_ips_ports")
    report["is_sm_ips_ports_values"] = sorted(int(v) for v in np.unique(sm))

    # negative values, constants
    report["negative_value_columns"] = [c for c in tab.numeric_names if (tab.column(c) < 0).any()]
    report["constant_columns"] = [c for c in tab.numeric_names if tab.column(c).std() == 0]

    # duplicate rows and label conflicts (identifier excluded)
    matrix = np.concatenate([tab.numeric, tab.nominal.astype(np.float64)], axis=1)
    _, inverse, group_counts = np.unique(matrix, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    report["duplicate_feature_vectors_excess_rows"] = int(n - group_counts.size)
    first_label = np.full(group_counts.size, -1)
    conflict = np.zeros(group_counts.size, dtype=bool)
    for label in range(data.N_CLASSES):
        present = np.zeros(group_counts.size, dtype=bool)
        present[inverse[tab.y == label]] = True
        conflict |= present & (first_label >= 0) & (first_label != label)
        first_label = np.where((first_label < 0) & present, label, first_label)
    report["feature_vectors_with_conflicting_labels"] = int(conflict.sum())
    report["rows_in_conflicting_feature_vectors"] = int(conflict[inverse].sum())
    # accuracy ceiling of any deterministic classifier on D: predict the majority label of every
    # distinct feature vector (resubstitution), the rows outside the majority label are unavoidable errors
    label_counts = np.zeros((group_counts.size, data.N_CLASSES), dtype=np.int64)
    np.add.at(label_counts, (inverse, tab.y), 1)
    report["resubstitution_accuracy_ceiling"] = float(label_counts.max(axis=1).sum() / n)
    # share of rows of a held-out fold with an identical feature vector in the training part
    first_fold = protocol.tuning_partitions(tab.y)[0]
    in_train = np.zeros(group_counts.size, dtype=bool)
    in_train[inverse[first_fold.train]] = True
    report["tuning_fold0_test_rows_with_training_duplicate_share"] = float(in_train[inverse[first_fold.test]].mean())
    seen_labels = np.zeros((group_counts.size, data.N_CLASSES), dtype=bool)
    seen_labels[inverse[first_fold.train], tab.y[first_fold.train]] = True
    test_has_label = seen_labels[inverse[first_fold.test], tab.y[first_fold.test]]
    report["tuning_fold0_test_rows_with_same_label_duplicate_share"] = float(test_has_label.mean())

    # correlation (Pearson, numeric features)
    corr = np.corrcoef(tab.numeric, rowvar=False)
    pairs = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr[i, j]) >= 0.9:
                pairs.append({"feature_1": tab.numeric_names[i], "feature_2": tab.numeric_names[j],
                              "correlation": float(corr[i, j])})
    pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False).to_csv(
        out / "high_correlation_pairs.csv", index=False)
    report["n_pairs_correlation_at_least_0.9"] = len(pairs)
    report["n_pairs_correlation_at_least_0.99"] = sum(abs(p["correlation"]) >= 0.99 for p in pairs)

    # scale, outliers and skewness
    rows = []
    for c in tab.numeric_names:
        x = tab.column(c)
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        outside = float(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean()) if iqr > 0 else 0.0
        row = {"feature": c, "median_abs": float(np.median(np.abs(x))), "minimum": float(x.min()),
               "maximum": float(x.max()), "outlier_share_iqr": outside, "skewness": skewness(x),
               "n_distinct": int(np.unique(x).size)}
        row["skewness_log1p"] = skewness(np.log1p(x)) if c in preprocess.LOG_COLUMNS and x.min() >= 0 else None
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "numeric_features.csv", index=False)
    logged = table[table["feature"].isin(preprocess.LOG_COLUMNS)]
    report["log_transformed_features"] = int(len(logged))
    report["skewness_before_log_min"] = float(logged["skewness"].min())
    report["skewness_before_log_max"] = float(logged["skewness"].max())
    report["skewness_after_log_max"] = float(logged["skewness_log1p"].max())
    report["median_abs_largest"] = [table.loc[table["median_abs"].idxmax(), "feature"],
                                    float(table["median_abs"].max())]
    report["median_abs_smallest_positive"] = [
        table.loc[table.loc[table["median_abs"] > 0, "median_abs"].idxmin(), "feature"],
        float(table.loc[table["median_abs"] > 0, "median_abs"].min())]
    report["outlier_share_iqr_range"] = [float(table["outlier_share_iqr"].min()),
                                         float(table["outlier_share_iqr"].max())]
    report["n_features_with_outliers_above_10_percent"] = int((table["outlier_share_iqr"] > 0.10).sum())

    with open(out / "summary.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
