"""04 · ¿Aporta la distancia exacta dentro del modelo completo? (lección 1 de S6E9)

En `01_original.py` la distancia exacta, sola, daba un AUC de 0,74131 frente a 0,72510 por tramos.
Aquí se mide si eso mejora la baseline (CV 0,95882 ± 0,00063), con los mismos folds y parámetros:

  A. baseline + target encoding del valor exacto de Flight Distance, sin fuga:
     - filas de validación y de test: tasa calculada con las filas de entrenamiento del fold;
     - filas de entrenamiento: tasa calculada con folds internos (nunca con su propia etiqueta).
  B. baseline + tasa y nº de filas de esa distancia en el dataset ORIGINAL (son otros datos: no hay fuga).

Uso: uv run python experiments/04_distancia_te.py
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
DATA, OUT = RAIZ / "data", RAIZ / "experiments" / "outputs"
SEED, N_FOLDS, TARGET, DIST, SUAVIZADO = 42, 5, "satisfaction", "Flight Distance", 20

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
orig = pd.read_csv(DATA / "original" / "data.csv")
y = train[TARGET].astype(int).to_numpy()
y_orig = orig[TARGET].astype(int).to_numpy()
FEATURES = [c for c in test.columns if c != "id"]
CATEGORICAL = train[FEATURES].select_dtypes(exclude="number").columns.tolist()
for col in CATEGORICAL:
    levels = sorted(set(train[col]) | set(test[col]))
    train[col] = pd.Categorical(train[col], categories=levels)
    test[col] = pd.Categorical(test[col], categories=levels)

folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, y))
base = json.loads((OUT / "02_baseline.json").read_text())

LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": SEED, "verbose": -1,
}


def tasa_suavizada(claves, etiquetas):
    """Tasa de satisfechos por clave, suavizada hacia la media global de esas filas."""
    media = etiquetas.mean()
    g = pd.DataFrame({"k": claves, "y": etiquetas}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + SUAVIZADO * media) / (g["count"] + SUAVIZADO), media


def frames_te(tr_idx, va_idx):
    d = train[DIST].to_numpy()
    te_tr = np.zeros(len(tr_idx))
    internos = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(tr_idx, y[tr_idx])
    for a, b in internos:  # a y b son posiciones dentro de tr_idx
        enc, media = tasa_suavizada(d[tr_idx[a]], y[tr_idx[a]])
        te_tr[b] = pd.Series(d[tr_idx[b]]).map(enc).fillna(media).to_numpy()
    enc, media = tasa_suavizada(d[tr_idx], y[tr_idx])
    X_tr = train[FEATURES].iloc[tr_idx].assign(dist_te=te_tr)
    X_va = train[FEATURES].iloc[va_idx].assign(dist_te=train[DIST].iloc[va_idx].map(enc).fillna(media).to_numpy())
    X_te = test[FEATURES].assign(dist_te=test[DIST].map(enc).fillna(media).to_numpy())
    return X_tr, X_va, X_te


enc_o, media_o = tasa_suavizada(orig[DIST].to_numpy(), y_orig)
n_o = orig[DIST].value_counts()
for df in (train, test):
    df["dist_orig_rate"] = df[DIST].map(enc_o).fillna(media_o)
    df["dist_orig_n"] = df[DIST].map(n_o).fillna(0)


def frames_orig(tr_idx, va_idx):
    cols = FEATURES + ["dist_orig_rate", "dist_orig_n"]
    return train[cols].iloc[tr_idx], train[cols].iloc[va_idx], test[cols]


def cv(nombre, get_frames):
    oof, test_pred, scores, arboles, inicio = np.zeros(len(train)), np.zeros(len(test)), [], [], time.time()
    for k, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va, X_te = get_frames(tr_idx, va_idx)
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, y[tr_idx]), num_boost_round=5000,
                            valid_sets=[lgb.Dataset(X_va, y[va_idx])],
                            callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_idx] = booster.predict(X_va)
        test_pred += booster.predict(X_te) / N_FOLDS
        scores.append(roc_auc_score(y[va_idx], oof[va_idx]))
        arboles.append(booster.best_iteration)
        print(f"  fold {k}: {scores[-1]:.5f} (baseline {base['fold_scores'][k]:.5f}, "
              f"dif {scores[-1] - base['fold_scores'][k]:+.5f}) árboles {booster.best_iteration}", flush=True)
    dif = np.array(scores) - np.array(base["fold_scores"])
    print(f"{nombre}: CV {np.mean(scores):.5f} ± {np.std(scores):.5f} | OOF {roc_auc_score(y, oof):.5f} | "
          f"dif media {dif.mean():+.5f} | folds que mejoran {(dif > 0).sum()}/{N_FOLDS} | {time.time() - inicio:.0f}s")
    return {"fold_scores": scores, "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores)),
            "oof": float(roc_auc_score(y, oof)), "trees": arboles}, oof, test_pred


print(f"baseline: CV {base['cv_mean']:.5f} ± {base['cv_std']:.5f}")
resultados = {}
for clave, nombre, fn in [("te", "A. + target encoding de la distancia exacta", frames_te),
                          ("orig", "B. + tasa y nº de filas de la distancia en el original", frames_orig)]:
    print(f"\n{nombre}:")
    resultados[clave], oof, test_pred = cv(nombre, fn)
    np.save(OUT / f"04_oof_{clave}.npy", oof)
    np.save(OUT / f"04_test_{clave}.npy", test_pred)
(OUT / "04_distancia_te.json").write_text(json.dumps(resultados, indent=2))
