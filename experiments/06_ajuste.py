"""06 · Parámetros de LightGBM, max_bin y un «profesor» entrenado con el original.

Mismos folds que el notebook. Cuatro preguntas:

  A. ¿Basta con subir max_bin para que LightGBM vea cada distancia exacta sin target encoding?
     (Por defecto agrupa cada columna numérica en 255 tramos; hay 3.474 distancias distintas.)
  B. ¿Aporta como columna la predicción de un modelo entrenado solo con el original («profesor»)?
     Idea publicada en el foro por Sachith7 y por Rugved Bane (discusiones 745908 y 745555).
  C. Búsqueda aleatoria de parámetros en el fold 0 (16 configuraciones), sobre el mejor modelo.
  D. La mejor configuración, en los 5 folds; y la misma con learning_rate 0,02.

El fold 0 se usa para elegir, así que en D se mira también la media de los folds 1-4, que no han
participado en la elección.

Uso: uv run python experiments/06_ajuste.py
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from playground_s6e10.cv import LGB_PARAMS, OUT, SEED, distance_te, load, make_folds, run_variant

train, test, orig, y, y_orig, FEATURES, CATEGORICAL, RATINGS = load()
folds = make_folds(train, y)
te = distance_te(train, test, orig, y, folds)
REF_RAW = json.loads((OUT / "02_baseline.json").read_text())["fold_scores"]
REF_TE = json.loads((OUT / "04_distancia_te.json").read_text())["te"]["fold_scores"]
X, X_test = train[FEATURES], test[FEATURES]
resultados = {}


def guardar(clave, res, oof=None, test_pred=None):
    resultados[clave] = res
    if oof is not None:
        np.save(OUT / f"06_oof_{clave}.npy", oof)
        np.save(OUT / f"06_test_{clave}.npy", test_pred)
    (OUT / "06_ajuste.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))


def sin_te(k, tr_idx, va_idx):
    return X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], X_test


def con_te(k, tr_idx, va_idx):
    return (X.iloc[tr_idx].assign(dist_te=te[k]["tr"]), y[tr_idx],
            X.iloc[va_idx].assign(dist_te=te[k]["va"]), X_test.assign(dist_te=te[k]["te"]))


print(f"referencias: sin TE {np.mean(REF_RAW):.5f} | con TE {np.mean(REF_TE):.5f}")

print("\nA1. Sin target encoding, max_bin 4095 (referencia: modelo sin TE):")
guardar("maxbin_sin_te", run_variant("A1", sin_te, y, folds, len(test), reference=REF_RAW, params={"max_bin": 4095})[0])
print("\nA2. Con target encoding, max_bin 4095 (referencia: modelo con TE):")
guardar("maxbin_con_te", run_variant("A2", con_te, y, folds, len(test), reference=REF_TE, params={"max_bin": 4095})[0])

# B. Profesor: el mismo LightGBM de la parte 7.5 del notebook, entrenado solo con el original
profesor = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
profesor.fit(orig[FEATURES], y_orig)
prof_train = profesor.predict(X, raw_score=True)
prof_test = profesor.predict(X_test, raw_score=True)


def con_profesor(k, tr_idx, va_idx):
    X_tr, y_tr, X_va, X_te = con_te(k, tr_idx, va_idx)
    return (X_tr.assign(orig_logit=prof_train[tr_idx]), y_tr,
            X_va.assign(orig_logit=prof_train[va_idx]), X_te.assign(orig_logit=prof_test))


print("\nB. Con target encoding + predicción del profesor (referencia: modelo con TE):")
guardar("profesor", *run_variant("B", con_profesor, y, folds, len(test), reference=REF_TE))

# C. Búsqueda aleatoria en el fold 0
ESPACIO = {
    "num_leaves": [15, 31, 63, 127, 255],
    "min_child_samples": [20, 50, 100, 200, 500],
    "feature_fraction": [0.4, 0.5, 0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 10.0, 50.0],
}
rng = np.random.default_rng(SEED)
print("\nC. Búsqueda aleatoria en el fold 0 (referencia: modelo con TE, fold 0):")


def solo_validacion(k, tr_idx, va_idx):
    X_tr, y_tr, X_va, _ = con_te(k, tr_idx, va_idx)
    return X_tr, y_tr, X_va, None


filas = []
for i in range(16):
    cfg = {p: v[int(rng.integers(len(v)))] for p, v in ESPACIO.items()}
    print(f" config {i}: {cfg}")
    res, _, _ = run_variant(f"config {i}", solo_validacion, y, folds, len(test), reference=REF_TE, params=cfg, only_folds=[0])
    filas.append({**cfg, "auc_fold0": res["fold_scores"][0], "dif": res["diff_mean"], "arboles": res["trees"][0]})
    pd.DataFrame(filas).to_csv(OUT / "06_busqueda.csv", index=False)
busqueda = pd.DataFrame(filas).sort_values("auc_fold0", ascending=False)
print(busqueda.round(5).to_string(index=False))
mejor = {p: type(ESPACIO[p][0])(busqueda.iloc[0][p]) for p in ESPACIO}
print(f"mejor configuración en el fold 0: {mejor}  (por defecto: { {p: LGB_PARAMS.get(p, 0.0) for p in ESPACIO} })")

print("\nD1. Mejor configuración, 5 folds (referencia: modelo con TE):")
res, oof, test_pred = run_variant("D1", con_te, y, folds, len(test), reference=REF_TE, params=mejor)
d = np.array([res["fold_scores"][k] - REF_TE[k] for k in range(1, 5)])
res["diff_mean_folds_1_4"] = float(d.mean())
print(f"    dif media en los folds 1-4 (no usados para elegir): {d.mean():+.5f}")
guardar("ajustado", {**res, "params": mejor}, oof, test_pred)

print("\nD2. Mejor configuración con learning_rate 0,02, 5 folds (referencia: modelo con TE):")
res, oof, test_pred = run_variant("D2", con_te, y, folds, len(test), reference=REF_TE, params={**mejor, "learning_rate": 0.02})
guardar("ajustado_lr002", {**res, "params": {**mejor, "learning_rate": 0.02}}, oof, test_pred)
