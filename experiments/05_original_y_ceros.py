"""05 · Lo que quedó «sin medir» en la versión 1: filas del original y ceros de las valoraciones.

Todo se compara con el mejor modelo del notebook (LightGBM + target encoding de la distancia exacta,
CV 0,96009), con los mismos folds y parámetros, y la diferencia se mira fold a fold.

  0. Control: el mismo modelo con otra semilla de LightGBM. Da el «suelo de ruido»: cuánto se mueve la
     CV sin cambiar nada. Una mejora menor que eso no cuenta.
  1. Añadir las 129.880 filas del original al entrenamiento de cada fold (nunca a la validación).
  2. Lo mismo, con una columna que marca de dónde viene cada fila.
  3. Los 0 de las valoraciones convertidos en ausentes (NaN).
  4. Las 13 valoraciones como categóricas (cada nota, su propia categoría).
  5. Una columna con el número de ceros de la fila.

Uso: uv run python experiments/05_original_y_ceros.py
"""

import json

import numpy as np
import pandas as pd

from playground_s6e10.cv import OUT, distance_te, load, make_folds, run_variant

train, test, orig, y, y_orig, FEATURES, CATEGORICAL, RATINGS = load()
folds = make_folds(train, y)
te = distance_te(train, test, orig, y, folds)
REF = json.loads((OUT / "04_distancia_te.json").read_text())["te"]["fold_scores"]
print(f"referencia (LightGBM + TE de la distancia): CV {np.mean(REF):.5f} ± {np.std(REF):.5f}")

X, X_test, X_orig = train[FEATURES], test[FEATURES], orig[FEATURES]


def base(k, tr_idx, va_idx):
    return (X.iloc[tr_idx].assign(dist_te=te[k]["tr"]), y[tr_idx],
            X.iloc[va_idx].assign(dist_te=te[k]["va"]), X_test.assign(dist_te=te[k]["te"]))


def con_original(marca):
    def frames(k, tr_idx, va_idx):
        X_tr, y_tr, X_va, X_te = base(k, tr_idx, va_idx)
        extra = X_orig.assign(dist_te=te[k]["orig"])
        if marca:
            X_tr, X_va, X_te, extra = (d.assign(is_original=v) for d, v in [(X_tr, 0), (X_va, 0), (X_te, 0), (extra, 1)])
        return pd.concat([X_tr, extra], ignore_index=True), np.r_[y_tr, y_orig], X_va, X_te
    return frames


def transformar(fn):
    def frames(k, tr_idx, va_idx):
        X_tr, y_tr, X_va, X_te = base(k, tr_idx, va_idx)
        return fn(X_tr), y_tr, fn(X_va), fn(X_te)
    return frames


def ceros_nan(d):
    d = d.copy()
    d[RATINGS] = d[RATINGS].where(d[RATINGS] != 0)
    return d


def valoraciones_categoricas(d):
    d = d.copy()
    for c in RATINGS:
        d[c] = pd.Categorical(d[c], categories=range(6))
    return d


def n_ceros(d):
    return d.assign(n_zero_ratings=(d[RATINGS] == 0).sum(axis=1))


VARIANTES = [
    ("control_semilla", "0. Control: misma receta, semilla 43", base, {"seed": 43}),
    ("original", "1. + filas del original en el entrenamiento", con_original(False), None),
    ("original_marca", "2. + filas del original, con columna is_original", con_original(True), None),
    ("ceros_nan", "3. Ceros de las valoraciones como NaN", transformar(ceros_nan), None),
    ("valoraciones_cat", "4. Valoraciones como categóricas", transformar(valoraciones_categoricas), None),
    ("n_ceros", "5. + número de ceros de la fila", transformar(n_ceros), None),
]

resultados = {}
for clave, nombre, frames, params in VARIANTES:
    print(f"\n{nombre}:")
    resultados[clave], oof, test_pred = run_variant(nombre, frames, y, folds, len(test), reference=REF, params=params)
    np.save(OUT / f"05_oof_{clave}.npy", oof)
    np.save(OUT / f"05_test_{clave}.npy", test_pred)
    (OUT / "05_original_y_ceros.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
