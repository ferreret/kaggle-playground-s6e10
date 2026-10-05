"""09 · ¿Se suman las tres mejoras pequeñas?

Por separado, sobre el modelo del notebook (CV 0,96009), cada una da unos +0,00025:
el «profesor» (06), los parámetros ajustados (06) y la mezcla con la red neuronal (07).
Aquí se juntan: LightGBM con target encoding + profesor + parámetros ajustados, y después
se mezcla con la red con embedding de la distancia.

Uso: uv run python experiments/09_combinado.py
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from playground_s6e10.cv import OUT, SEED, SUBS, TARGET, distance_te, load, make_folds, run_variant

train, test, orig, y, y_orig, FEATURES, CATEGORICAL, RATINGS = load()
folds = make_folds(train, y)
te = distance_te(train, test, orig, y, folds)
REF = json.loads((OUT / "04_distancia_te.json").read_text())["te"]["fold_scores"]
AJUSTADOS = json.loads((OUT / "06_ajuste.json").read_text())["ajustado"]["params"]
X, X_test = train[FEATURES], test[FEATURES]

profesor = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
profesor.fit(orig[FEATURES], y_orig)
prof_train, prof_test = profesor.predict(X, raw_score=True), profesor.predict(X_test, raw_score=True)


def frames(k, tr_idx, va_idx):
    return (X.iloc[tr_idx].assign(dist_te=te[k]["tr"], orig_logit=prof_train[tr_idx]), y[tr_idx],
            X.iloc[va_idx].assign(dist_te=te[k]["va"], orig_logit=prof_train[va_idx]),
            X_test.assign(dist_te=te[k]["te"], orig_logit=prof_test))


print(f"referencia (modelo del notebook): CV {np.mean(REF):.5f} | parámetros ajustados: {AJUSTADOS}", flush=True)
print("\nLightGBM + TE + profesor + parámetros ajustados:")
res, oof, test_pred = run_variant("combinado", frames, y, folds, len(test), reference=REF, params=AJUSTADOS)
np.save(OUT / "09_oof_combinado.npy", oof)
np.save(OUT / "09_test_combinado.npy", test_pred)

red_oof, red_test = np.load(OUT / "07_oof_con_distancia.npy"), np.load(OUT / "07_test_con_distancia.npy")
print("\nmezcla con la red (peso de la red -> CV, dif sobre el modelo del notebook, dif sobre el combinado solo):")
base = np.array([res["fold_scores"][k] for k in range(5)])
mezcla = {}
for w in [0.0, 0.1, 0.2, 0.3, 0.4]:
    f = np.array([roc_auc_score(y[va], w * red_oof[va] + (1 - w) * oof[va]) for _, va in folds])
    mezcla[w] = {"cv": float(f.mean()), "dif_notebook": float(f.mean() - np.mean(REF)), "dif_combinado": float((f - base).mean()),
                 "folds_mejoran_combinado": int((f > base).sum()), "peor_fold_combinado": float((f - base).min())}
    print(f"  {w:.1f} -> {f.mean():.5f}  {f.mean() - np.mean(REF):+.5f}  {(f - base).mean():+.5f} "
          f"({(f > base).sum()}/5, peor fold {(f - base).min():+.5f})")
(OUT / "09_combinado.json").write_text(json.dumps({"combinado": res, "mezcla": mezcla}, indent=2))

for nombre, pred in [("03_lgbm_combinado", test_pred), ("04_mezcla_red_20", 0.2 * red_test + 0.8 * test_pred)]:
    pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(SUBS / f"{nombre}.csv", index=False)
    print(f"envío generado (sin subir): submissions/{nombre}.csv")
