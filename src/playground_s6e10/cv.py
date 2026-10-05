"""Shared CV harness for the experiments: same data, folds, parameters and encoding as the notebook.

Every experiment compares a variant against the notebook's best model (LightGBM + fold-safe target
encoding of the exact flight distance) on the same five folds, so differences are paired fold by fold.
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT, SUBS = ROOT / "data", ROOT / "experiments" / "outputs", ROOT / "submissions"

SEED, N_FOLDS, SMOOTHING = 42, 5, 20
TARGET, DIST = "satisfaction", "Flight Distance"

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "verbose": -1,
}


def load():
    """Return train, test, original, y, y_orig, FEATURES, CATEGORICAL, RATINGS with shared category codes."""
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    orig = pd.read_csv(DATA / "original" / "data.csv")
    y = train[TARGET].astype(int).to_numpy()
    y_orig = orig[TARGET].astype(int).to_numpy()
    features = [c for c in test.columns if c != "id"]
    categorical = train[features].select_dtypes(exclude="number").columns.tolist()
    ratings = [c for c in features if c not in categorical and train[c].max() <= 5]
    for col in categorical:
        levels = sorted(set(train[col]) | set(test[col]) | set(orig[col]))
        for df in (train, test, orig):
            df[col] = pd.Categorical(df[col], categories=levels)
    return train, test, orig, y, y_orig, features, categorical, ratings


def make_folds(train, y):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, y))


def smoothed_rate(keys, labels):
    """Satisfaction rate per key, pulled towards the overall rate of those rows."""
    prior = labels.mean()
    g = pd.DataFrame({"k": keys, "y": labels}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + SMOOTHING * prior) / (g["count"] + SMOOTHING), prior


def distance_te(train, test, orig, y, folds):
    """Fold-safe target encoding of the exact distance, computed once per fold.

    Returns one dict per fold with the encoded column for the training rows (inner folds), the
    validation rows, the test rows and the original rows (the last three use the fold's training rows).
    """
    dist = train[DIST].to_numpy()
    cache = []
    for tr_idx, va_idx in folds:
        te_tr = np.zeros(len(tr_idx))
        inner = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(tr_idx, y[tr_idx])
        for a, b in inner:
            enc, prior = smoothed_rate(dist[tr_idx[a]], y[tr_idx[a]])
            te_tr[b] = pd.Series(dist[tr_idx[b]]).map(enc).fillna(prior).to_numpy()
        enc, prior = smoothed_rate(dist[tr_idx], y[tr_idx])
        cache.append({
            "tr": te_tr,
            "va": train[DIST].iloc[va_idx].map(enc).fillna(prior).to_numpy(),
            "te": test[DIST].map(enc).fillna(prior).to_numpy(),
            "orig": orig[DIST].map(enc).fillna(prior).to_numpy(),
        })
    return cache


def run_variant(name, get_frames, y, folds, n_test, reference=None, params=None, only_folds=None):
    """Run one variant through the folds.

    get_frames(k, tr_idx, va_idx) -> (X_tr, y_tr, X_va, X_te). `reference` is the list of per-fold
    scores to compare against. Returns a dict with scores, trees, OOF and averaged test predictions.
    """
    params = {**LGB_PARAMS, **(params or {})}
    oof, test_pred, scores, trees, start = np.full(len(y), np.nan), np.zeros(n_test), {}, {}, time.time()
    use = range(len(folds)) if only_folds is None else only_folds
    for k in use:
        tr_idx, va_idx = folds[k]
        X_tr, y_tr, X_va, X_te = get_frames(k, tr_idx, va_idx)
        booster = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=20000,
                            valid_sets=[lgb.Dataset(X_va, y[va_idx])],
                            callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_idx] = booster.predict(X_va)
        if X_te is not None:
            test_pred += booster.predict(X_te) / len(use)
        scores[k] = roc_auc_score(y[va_idx], oof[va_idx])
        trees[k] = booster.best_iteration
        ref = "" if reference is None else f" (ref {reference[k]:.5f}, diff {scores[k] - reference[k]:+.5f})"
        print(f"  fold {k}: {scores[k]:.5f}{ref} trees {trees[k]}", flush=True)
    s = np.array(list(scores.values()))
    line = f"{name}: CV {s.mean():.5f} ± {s.std():.5f}"
    result = {"name": name, "fold_scores": {int(k): float(v) for k, v in scores.items()},
              "cv_mean": float(s.mean()), "cv_std": float(s.std()), "trees": {int(k): int(v) for k, v in trees.items()},
              "seconds": round(time.time() - start)}
    if reference is not None:
        d = s - np.array([reference[k] for k in scores])
        line += f" | mean diff {d.mean():+.5f} (min {d.min():+.5f}, max {d.max():+.5f}) | folds better {(d > 0).sum()}/{len(d)}"
        result.update(diff_mean=float(d.mean()), diff_min=float(d.min()), diff_max=float(d.max()),
                      folds_better=int((d > 0).sum()))
    print(f"{line} | {result['seconds']}s", flush=True)
    return result, oof, test_pred
