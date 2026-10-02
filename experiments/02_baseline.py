"""02 · Folds estratificados y baseline LightGBM (paso 2 del día 1-2).

Mismo código que el notebook (run_cv, LGB_PARAMS, make_lgbm), para que la cifra que salga
aquí sea la que saldrá allí. Guarda las predicciones fuera de fold y de test para reutilizarlas
(gráficas, comparación con las features del día 2+) y el fichero del primer envío.

Uso: uv run python experiments/02_baseline.py
Salidas (ignoradas por git): experiments/outputs/02_*.{npy,csv,json} y submissions/01_lgbm_raw.csv
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RAIZ = Path(__file__).resolve().parents[1]
DATA, OUT, SUBS = RAIZ / "data", RAIZ / "experiments" / "outputs", RAIZ / "submissions"
OUT.mkdir(exist_ok=True)
SUBS.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5
TARGET = "satisfaction"

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train[TARGET].astype(int).to_numpy()

FEATURES = [c for c in test.columns if c != "id"]
CATEGORICAL = train[FEATURES].select_dtypes(exclude="number").columns.tolist()

print(f"train: {train.shape}   test: {test.shape}   satisfechos: {y.mean():.6f}")
print("versiones -> pandas", pd.__version__, "| lightgbm", lgb.__version__)


def score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, y))
print("\nfolds:")
for k, (tr_idx, va_idx) in enumerate(folds):
    print(f"  fold {k}: train {len(tr_idx):,} | validación {len(va_idx):,} | "
          f"satisfechos en validación {y[va_idx].mean():.6f}")

RESULTS = {}
FOLD_SCORES = {}


def run_cv(name, fit_predict, keep_first_fold=False):
    # fit_predict(tr_idx, va_idx) -> (validation predictions, test predictions, info string, extras)
    oof, test_pred, scores = np.zeros(len(train)), np.zeros(len(test)), []
    start, first_fold = time.time(), None
    for k, (tr_idx, va_idx) in enumerate(folds):
        va_pred, te_pred, info, extras = fit_predict(tr_idx, va_idx)
        oof[va_idx] = va_pred
        test_pred += te_pred / N_FOLDS
        scores.append(score(y[va_idx], va_pred))
        if keep_first_fold and k == 0:
            first_fold = extras
        print(f"  fold {k}: {scores[-1]:.5f} {info}", flush=True)
    RESULTS[name] = {"cv_mean": np.mean(scores), "cv_std": np.std(scores), "oof": score(y, oof)}
    FOLD_SCORES[name] = scores
    print(f"{name}: CV {np.mean(scores):.5f} ± {np.std(scores):.5f} | OOF {RESULTS[name]['oof']:.5f} "
          f"| {time.time() - start:.0f}s")
    return test_pred, first_fold, oof


for col in CATEGORICAL:
    levels = sorted(set(train[col]) | set(test[col]))  # same codes in train and test
    train[col] = pd.Categorical(train[col], categories=levels)
    test[col] = pd.Categorical(test[col], categories=levels)

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

ARBOLES = []
GANANCIA = []


def make_lgbm(get_frames):
    # get_frames(tr_idx, va_idx) -> (X_tr, X_va, X_te) with the columns to use
    def fit_predict(tr_idx, va_idx):
        X_tr, X_va, X_te = get_frames(tr_idx, va_idx)
        booster = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X_tr, y[tr_idx]),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(X_va, y[va_idx])],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        ARBOLES.append(booster.best_iteration)
        GANANCIA.append(booster.feature_importance("gain", iteration=booster.best_iteration))
        return (booster.predict(X_va), booster.predict(X_te),
                f"(trees: {booster.best_iteration})", (booster, X_va))
    return fit_predict


def raw_frames(tr_idx, va_idx):
    return train[FEATURES].iloc[tr_idx], train[FEATURES].iloc[va_idx], test[FEATURES]


print("\nLightGBM (raw features):")
NOMBRE = "LightGBM (raw features)"
test_pred, _, oof = run_cv(NOMBRE, make_lgbm(raw_frames))

# Referencia trivial: la mejor variable sola (cuánto hay que superar)
print("\nAUC de cada variable numérica por sí sola (sin modelo, valor crudo):")
unis = {c: roc_auc_score(y, train[c].fillna(train[c].median())) for c in FEATURES if c not in CATEGORICAL}
print(pd.Series(unis).sort_values(ascending=False).round(5).to_string())

np.save(OUT / "02_oof_lgbm_raw.npy", oof)
np.save(OUT / "02_test_lgbm_raw.npy", test_pred)
ganancia = pd.Series(np.mean(GANANCIA, axis=0), index=FEATURES).sort_values(ascending=False)
ganancia.rename("gain").to_csv(OUT / "02_gain_lgbm_raw.csv")
print("\nimportancia por ganancia (media de los 5 folds, % del total):")
print((ganancia / ganancia.sum() * 100).round(2).to_string())

(OUT / "02_baseline.json").write_text(json.dumps({
    "model": NOMBRE,
    "fold_scores": FOLD_SCORES[NOMBRE],
    "cv_mean": RESULTS[NOMBRE]["cv_mean"],
    "cv_std": RESULTS[NOMBRE]["cv_std"],
    "oof": RESULTS[NOMBRE]["oof"],
    "trees": ARBOLES,
    "params": LGB_PARAMS,
}, indent=2))

submission = pd.DataFrame({"id": test["id"], TARGET: test_pred})
submission.to_csv(SUBS / "01_lgbm_raw.csv", index=False)
print(f"\nenvío: {SUBS / '01_lgbm_raw.csv'}  ({len(submission):,} filas, "
      f"media {test_pred.mean():.6f}, min {test_pred.min():.6f}, max {test_pred.max():.6f})")
