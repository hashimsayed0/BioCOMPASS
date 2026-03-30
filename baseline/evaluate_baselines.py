#!/usr/bin/env python3
"""
Evaluate baseline immune scoring methods and ML models on ICI response prediction.

Supports split settings:
  loco  — leave-one-cohort-out      (group = Dataset / cohort name)
  locto — leave-one-cancer-type-out (group = TCGA_Study)
  loto  — leave-one-ICI-target-out  (group = ICI_Target string)

For each immune scoring method:
  1. Score all samples with the method (unsupervised).
  2. Fit a LogisticRegression on train-split scores.
  3. Predict on test-split; compute AUC.
  Raw score AUC (no training) is also reported.

Also evaluates sklearn ML models (LogReg, RF, GBM) trained on train split and
evaluated on test split using:
  - log2(TPM+1) gene expression (PCA-reduced, COMPASS gene vocabulary)
  - Pre-computed clinical / biomarker features
"""

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
# Patch sklearn with MKL/DAAL for CPU acceleration via Intel sklearnex if available
try:
    from sklearnex import patch_sklearn
    patch_sklearn(verbose=False)
except ImportError:
    pass

from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GENE_EXP_PATH   = DATA / "gene_exp.tsv"
CLINICAL_PATH   = DATA / "clinical_features.tsv"
GENE_TOKEN_PATH = ROOT / "compass" / "tokenizer" / "gene_tokens_long.json"

sys.path.insert(0, str(ROOT))


def _load_and_filter(clinical_file, all_cohorts=None):
    df = pd.read_csv(clinical_file, sep='\t', index_col='Run_ID')
    df = df[df['Responder'].notna() & (df['Sample_Treatment'] == 'Pre')]
    if all_cohorts is not None:
        df = df[df['Dataset'].isin(all_cohorts)]
    return df


def discover_cohorts(clinical_file, all_cohorts=None):
    df = _load_and_filter(clinical_file, all_cohorts)
    return sorted(df['Dataset'].unique()), df['Dataset'], df


def discover_cancer_types(clinical_file, all_cohorts=None):
    df = _load_and_filter(clinical_file, all_cohorts)
    return sorted(df['TCGA_Study'].dropna().unique()), df['TCGA_Study'], df


def discover_ici_target_groups(clinical_file, all_cohorts=None):
    """Discover ICI target groups using ICI_Target from clinical features (LOTO)."""
    df = _load_and_filter(clinical_file, all_cohorts)
    group_assignments = df['ICI_Target'].fillna('unknown')
    return sorted(group_assignments.unique().tolist()), group_assignments, df

DEFAULT_COHORTS = [
    "Gide_Cell_2019",
    "HugoLo_IPRES_2016",
    "IMmotion150",
    "IMVigor210",
    "Kim_NatMed_2018",
    "Liu_NatMed_2019",
    "Riaz_Nivolumab_2017",
    "VanAllen_antiCTLA4_2015",
]

# Per-cohort cancer_type / drug_target for immune score method init.
# cancer_type: TCGA study code.
# drug_target: Kong_NetBio valid values — PD1, CTLA4, PDL1, PD1_CTLA4, PD1_PDL1_CTLA4
COHORT_META = {
    "Gide_Cell_2019":          {"cancer_type": "SKCM", "drug_target": "PD1_CTLA4"},
    "HugoLo_IPRES_2016":       {"cancer_type": "SKCM", "drug_target": "PD1"},
    "IMmotion150":             {"cancer_type": "KIRC", "drug_target": "PDL1"},
    "IMVigor210":              {"cancer_type": "BLCA", "drug_target": "PDL1"},
    "Kim_NatMed_2018":         {"cancer_type": "STAD", "drug_target": "PD1"},
    "Liu_NatMed_2019":         {"cancer_type": "SKCM", "drug_target": "PD1"},
    "Riaz_Nivolumab_2017":     {"cancer_type": "SKCM", "drug_target": "PD1"},
    "VanAllen_antiCTLA4_2015": {"cancer_type": "SKCM", "drug_target": "CTLA4"},
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_compass_genes():
    import json
    with open(GENE_TOKEN_PATH) as f:
        token_map = json.load(f)
    return {v for k, v in token_map.items() if int(k) >= 0}


def load_clinical(sample_ids):
    df = pd.read_csv(CLINICAL_PATH, sep="\t", index_col=0)
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    df = df[num_cols + bool_cols].copy()
    for c in bool_cols:
        df[c] = df[c].astype(float)
    return df.loc[df.index.intersection(sample_ids)]


def load_gene_exp(sample_ids, compass_genes):
    """Load gene_exp.tsv, keeping only COMPASS-vocabulary gene columns."""
    print("  [gene_exp] reading header …", flush=True)
    with open(GENE_EXP_PATH) as fh:
        all_genes = fh.readline().rstrip("\n").split("\t")[1:]
    keep_cols = [i + 1 for i, g in enumerate(all_genes) if g in compass_genes]
    print(f"  [gene_exp] COMPASS genes: {len(keep_cols)} / {len(all_genes)}", flush=True)

    print("  [gene_exp] loading …", flush=True)
    df = pd.read_csv(GENE_EXP_PATH, sep="\t", index_col=0, usecols=[0] + keep_cols)
    df = df.loc[df.index.intersection(sample_ids)]
    print(f"  [gene_exp] {df.shape[0]} samples × {df.shape[1]} genes", flush=True)
    return df


# ── utilities ─────────────────────────────────────────────────────────────────

NAN_METRICS = {"AUC": np.nan, "accuracy": np.nan, "precision": np.nan,
               "recall": np.nan, "f1": np.nan}


def find_best_threshold(y_train, proba_train):
    """Find optimal threshold via Youden's J on train set predictions."""
    fpr, tpr, thresholds = roc_curve(y_train, proba_train)
    return thresholds[np.argmax(tpr - fpr)]


def compute_metrics(y_true, proba, threshold=0.5):
    """Compute AUC, accuracy, precision, recall, F1 from predicted probabilities.

    threshold should be derived from the train set (via find_best_threshold) to
    avoid test-set leakage.
    """
    try:
        if len(np.unique(y_true)) < 2:
            return NAN_METRICS.copy()
        auc = roc_auc_score(y_true, proba)
        pred = (proba >= threshold).astype(int)
        return {
            "AUC":       auc,
            "accuracy":  accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall":    recall_score(y_true, pred, zero_division=0),
            "f1":        f1_score(y_true, pred, zero_division=0),
        }
    except Exception:
        return NAN_METRICS.copy()


def logreg_on_scores(s_train, y_train, s_test, y_test, seed=42):
    """Fit 1-feature LogReg on train immune scores, evaluate on test."""
    mask_tr = ~np.isnan(s_train)
    mask_te = ~np.isnan(s_test)
    if mask_tr.sum() < 4 or len(np.unique(y_train[mask_tr])) < 2:
        return NAN_METRICS.copy()
    if mask_te.sum() < 2 or len(np.unique(y_test[mask_te])) < 2:
        return NAN_METRICS.copy()
    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=1000, random_state=seed))])
    clf.fit(s_train[mask_tr].reshape(-1, 1), y_train[mask_tr])
    proba_train = clf.predict_proba(s_train[mask_tr].reshape(-1, 1))[:, 1]
    threshold = find_best_threshold(y_train[mask_tr], proba_train)
    proba_test = clf.predict_proba(s_test[mask_te].reshape(-1, 1))[:, 1]
    return compute_metrics(y_test[mask_te], proba_test, threshold=threshold)


def sklearn_train_test(X_train, y_train, X_test, y_test, feature_type, seed=42, n_pca=50):
    """Fit LogReg/RF/GBM on train, evaluate on test. Returns {model_name: metrics_dict}.

    For gene expression, runs models on both PCA-reduced (50 components) and full
    log2(TPM+1) features. sklearnex CPU (MKL/DAAL) acceleration is used if installed.
    """
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)
    const   = X_train.columns[X_train.std() == 0]
    X_train = X_train.drop(columns=const)
    X_test  = X_test.drop(columns=const, errors="ignore")
    if X_train.shape[1] == 0:
        return {}

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {}

    def _fit_eval(pipe, X_tr, X_te):
        pipe.fit(X_tr, y_train)
        proba_train = pipe.predict_proba(X_tr)[:, 1]
        proba_test  = pipe.predict_proba(X_te)[:, 1]
        threshold = find_best_threshold(y_train, proba_train)
        return compute_metrics(y_test, proba_test, threshold=threshold)

    results = {}

    if feature_type == "geneexp":
        X_log_tr = np.log2(X_train.values + 1)
        X_log_te = np.log2(X_test.values + 1)

        # ── PCA-reduced branch (LogReg, RF, GBM) ─────────────────────────
        n_comp = min(n_pca, X_log_tr.shape[0] - 1, X_log_tr.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        X_pca_tr = pca.fit_transform(X_log_tr)
        X_pca_te = pca.transform(X_log_te)

        pca_models = {
            "LogReg_PCA": LogisticRegression(max_iter=1000, random_state=seed),
            "RF_PCA":     RandomForestClassifier(n_estimators=100, random_state=seed),
            "GBM_PCA":    GradientBoostingClassifier(n_estimators=100, random_state=seed),
        }
        for name, clf in pca_models.items():
            print(f"      fitting {name} …", flush=True)
            try:
                results[name] = _fit_eval(
                    Pipeline([("sc", StandardScaler()), ("clf", clf)]),
                    X_pca_tr, X_pca_te)
            except Exception as e:
                print(f"      {name} ERROR: {e}", flush=True)
                results[name] = NAN_METRICS.copy()

        # ── Full-feature (saga LogReg, RF; sklearnex-accelerated) ────────
        full_models = {
            "LogReg_full": LogisticRegression(solver="saga", max_iter=500,
                                              C=0.1, random_state=seed),
            "RF_full":     RandomForestClassifier(n_estimators=100, random_state=seed),
        }
        for name, clf in full_models.items():
            print(f"      fitting {name} …", flush=True)
            try:
                results[name] = _fit_eval(
                    Pipeline([("sc", StandardScaler()), ("clf", clf)]),
                    X_log_tr, X_log_te)
            except Exception as e:
                print(f"      {name} ERROR: {e}", flush=True)
                results[name] = NAN_METRICS.copy()

    else:
        X_tr = X_train.values
        X_te = X_test.values
        models = {
            "LogReg": LogisticRegression(max_iter=1000, random_state=seed),
            "RF":     RandomForestClassifier(n_estimators=100, random_state=seed),
            "GBM":    GradientBoostingClassifier(n_estimators=100, random_state=seed),
        }
        for name, clf in models.items():
            print(f"      fitting {name} …", flush=True)
            try:
                results[name] = _fit_eval(
                    Pipeline([("sc", StandardScaler()), ("clf", clf)]),
                    X_tr, X_te)
            except Exception as e:
                print(f"      {name} ERROR: {e}", flush=True)
                results[name] = NAN_METRICS.copy()

    return results


def get_cohort_meta(sample_ids, df_info):
    """Pick the dominant cancer_type and drug_target for a set of test samples."""
    # Use most frequent TCGA_Study and ICI_Target in this group
    sub = df_info.loc[df_info.index.intersection(sample_ids)]
    if sub.empty:
        return {"cancer_type": "SKCM", "drug_target": "PD1"}

    cancer_type = sub["TCGA_Study"].value_counts().index[0] \
        if "TCGA_Study" in sub.columns else "SKCM"

    # Map ICI_Target → Kong_NetBio drug_target token
    ici_target = sub["ICI_Target"].value_counts().index[0] \
        if "ICI_Target" in sub.columns else "PD1"
    target_map = {
        "PD1": "PD1", "PD-1": "PD1",
        "PD-L1": "PDL1", "PDL1": "PDL1",
        "CTLA4": "CTLA4", "CTLA-4": "CTLA4",
        "CTLA4 + PD1": "PD1_CTLA4", "PD1 + CTLA4": "PD1_CTLA4",
    }
    drug_target = target_map.get(ici_target, "PD1")
    return {"cancer_type": cancer_type, "drug_target": drug_target}


# ── main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(all_groups, group_assignments, df_info,
                   gene_exp_df, clinical_df, labels_series,
                   groups_to_test, setting, seed=42):
    from baseline.immnue_score import immnue_score_methods

    all_results = []

    for test_group in groups_to_test:
        train_groups = [g for g in all_groups if g != test_group]

        test_ids  = group_assignments[group_assignments == test_group].index.tolist()
        train_ids = group_assignments[group_assignments.isin(train_groups)].index.tolist()

        # Align to samples we actually have labels for
        test_ids  = [i for i in test_ids  if i in labels_series.index]
        train_ids = [i for i in train_ids if i in labels_series.index]

        y_test  = labels_series.loc[test_ids].values.astype(int)
        y_train = labels_series.loc[train_ids].values.astype(int)

        n_pos = y_test.sum()
        n_neg = (y_test == 0).sum()
        print(f"\n{'='*60}")
        print(f"[{setting.upper()}] Test group: {test_group}")
        print(f"  Test  : {len(test_ids)} samples  ({n_pos} R / {n_neg} NR)")
        print(f"  Train : {len(train_ids)} samples from {len(train_groups)} groups")
        print('='*60)

        if n_pos < 2 or n_neg < 2:
            print("  Skipping — too few classes in test set.")
            continue

        # Derive cancer_type / drug_target from actual test-group sample metadata
        meta = get_cohort_meta(test_ids, df_info)
        print(f"  cancer_type={meta['cancer_type']}  drug_target={meta['drug_target']}")

        # ── 1. Immune score baselines + LogReg ────────────────────────────
        ge_test  = gene_exp_df.index.intersection(test_ids).tolist()
        ge_train = gene_exp_df.index.intersection(train_ids).tolist()

        if ge_test and ge_train:
            print(f"\n  [Immune Score Baselines + LogReg]")
            df_train_ge = gene_exp_df.loc[ge_train]
            df_test_ge  = gene_exp_df.loc[ge_test]
            y_tr_ge = labels_series.loc[ge_train].values.astype(int)
            y_te_ge = labels_series.loc[ge_test].values.astype(int)

            for method_name, MethodClass in immnue_score_methods.items():
                try:
                    method = MethodClass(cancer_type=meta["cancer_type"],
                                        drug_target=meta["drug_target"])
                    # fit on train only, transform train and test separately
                    method.fit(df_train_ge, seed=seed)
                    s_train = method.transform(df_train_ge).iloc[:, 0].values.astype(float)
                    s_test  = method.transform(df_test_ge).iloc[:, 0].values.astype(float)

                    metrics = logreg_on_scores(s_train, y_tr_ge, s_test, y_te_ge, seed=seed)

                    print(f"    {method_name:15s}  AUC={metrics['AUC']:.3f}  "
                          f"acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}")
                    all_results.append({"test_group": str(test_group), "setting": setting,
                                        "method_type": "immune_score_logreg",
                                        "method": method_name, **metrics})
                except Exception as e:
                    print(f"    {method_name:15s}  ERROR: {e}")
                    all_results.append({"test_group": str(test_group), "setting": setting,
                                        "method_type": "immune_score_logreg",
                                        "method": method_name, **NAN_METRICS})

        # ── 2. ML on gene expression ──────────────────────────────────────
        if ge_test and ge_train and len(np.unique(y_tr_ge)) > 1 and len(np.unique(y_te_ge)) > 1:
            print(f"\n  [ML on Gene Expression (train→test)]")
            res = sklearn_train_test(gene_exp_df.loc[ge_train], y_tr_ge,
                                     gene_exp_df.loc[ge_test],  y_te_ge, "geneexp", seed=seed)
            for mname, metrics in res.items():
                print(f"    {mname:10s}  AUC={metrics['AUC']:.3f}  "
                      f"acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}")
                all_results.append({"test_group": str(test_group), "setting": setting,
                                    "method_type": "sklearn_geneexp",
                                    "method": mname, **metrics})

        # ── 3. ML on clinical biomarkers ──────────────────────────────────
        cl_test  = clinical_df.index.intersection(test_ids).tolist()
        cl_train = clinical_df.index.intersection(train_ids).tolist()
        y_tr_cl  = labels_series.loc[cl_train].values.astype(int)
        y_te_cl  = labels_series.loc[cl_test].values.astype(int)

        if (cl_test and cl_train
                and len(np.unique(y_tr_cl)) > 1 and len(np.unique(y_te_cl)) > 1):
            print(f"\n  [ML on Clinical Biomarkers (train→test)]")
            res = sklearn_train_test(clinical_df.loc[cl_train], y_tr_cl,
                                     clinical_df.loc[cl_test],  y_te_cl, "clinical", seed=seed)
            for mname, metrics in res.items():
                print(f"    {mname:10s}  AUC={metrics['AUC']:.3f}  "
                      f"acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}")
                all_results.append({"test_group": str(test_group), "setting": setting,
                                    "method_type": "sklearn_clinical",
                                    "method": mname, **metrics})

    return pd.DataFrame(all_results)


# ── entry point ───────────────────────────────────────────────────────────────

def main(args):
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cohorts_filter = (
        [c.strip() for c in args.all_cohorts.split(",")]
        if args.all_cohorts else DEFAULT_COHORTS
    )

    # ── discover groups using the same logic as tune_test.py ─────────────
    print(f"Setting: {args.setting.upper()}")
    clinical_file = str(CLINICAL_PATH)

    if args.setting == "loco":
        all_groups, group_assignments, df_info = \
            discover_cohorts(clinical_file, all_cohorts_filter)
    elif args.setting == "locto":
        all_groups, group_assignments, df_info = \
            discover_cancer_types(clinical_file, all_cohorts_filter)
    elif args.setting == "loto":
        all_groups, group_assignments, df_info = \
            discover_ici_target_groups(clinical_file, all_cohorts_filter)

    group_assignments = group_assignments.reindex(df_info.index)

    # ── parse --groups filter ─────────────────────────────────────────────
    if args.groups:
        raw = [g.strip() for g in args.groups.split(",")]
        selected_groups = raw
    else:
        selected_groups = all_groups

    # ── parse --test_group ────────────────────────────────────────────────
    if args.test_group:
        tg = args.test_group
        groups_to_test = [tg]
    else:
        groups_to_test = selected_groups

    print(f"Groups pool : {selected_groups}")
    print(f"Groups to test: {groups_to_test}")

    # ── load data ─────────────────────────────────────────────────────────
    all_sample_ids = df_info.index.tolist()

    print("\nLoading COMPASS gene vocabulary …")
    compass_genes = load_compass_genes()
    print(f"  {len(compass_genes)} genes")

    print("Loading clinical features …")
    clinical_df = load_clinical(all_sample_ids)

    print("Loading gene expression …")
    gene_exp_df = load_gene_exp(all_sample_ids, compass_genes)

    # Build labels Series (int 0/1)
    labels_series = df_info["Responder"].map({True: 1, False: 0, 1: 1, 0: 0}).dropna().astype(int)

    # ── run ───────────────────────────────────────────────────────────────
    results_df = run_evaluation(
        all_groups=selected_groups,
        group_assignments=group_assignments,
        df_info=df_info,
        gene_exp_df=gene_exp_df,
        clinical_df=clinical_df,
        labels_series=labels_series,
        groups_to_test=groups_to_test,
        setting=args.setting,
        seed=args.seed,
    )

    # ── save ──────────────────────────────────────────────────────────────
    run_dir = out_dir / args.setting
    run_dir.mkdir(parents=True, exist_ok=True)

    out_file = run_dir / f"baseline_results_seed{args.seed}.tsv"
    results_df.to_csv(out_file, sep="\t", index=False)
    print(f"\nResults saved to {out_file}")

    # Print summary to console only
    metric_cols = ["AUC", "accuracy", "precision", "recall", "f1"]
    summary = (results_df
               .groupby(["method_type", "method"])[metric_cols]
               .mean()
               .add_prefix("mean_")
               .sort_values("mean_AUC", ascending=False))
    print("\nTop methods by mean AUC:")
    print(summary[["mean_AUC", "mean_accuracy", "mean_f1"]].head(20).to_string())

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate baseline methods with leave-one-group-out CV."
    )
    parser.add_argument("--setting", type=str, default="loco",
                        choices=["loco", "loto", "locto"],
                        help="Split setting: loco (cohort), locto (cancer type), "
                             "loto (ICI target) (default: loco)")
    parser.add_argument("--all_cohorts", type=str, default=None,
                        help="Comma-separated cohort names to restrict data pool "
                             "(default: all 8 cohorts)")
    parser.add_argument("--groups", type=str, default=None,
                        help="Comma-separated groups to include in the pool "
                             "(subset of all_cohorts groups after setting-based splitting)")
    parser.add_argument("--test_group", type=str, default=None,
                        help="Run only this single test fold")
    parser.add_argument("--output_dir", default="baseline/results",
                        help="Directory to save results (default: baseline/results)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    main(args)
