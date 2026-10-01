"""01 · El dataset original frente al sintético (lección 1 de S6E9).

Original: arseniyshutko/binary-aviation-satisfaction-129k (data/original/data.csv).
Preguntas:
  1. ¿Mismo esquema, mismo balance, mismos ausentes y ceros?
  2. ¿El generador copió valores exactos (Flight Distance, retrasos, edad)?
  3. ¿Esos valores exactos llevan target asociado, más allá de la tendencia suave?
  4. ¿Hay filas enteras del original repetidas en train o test?

Uso: uv run python experiments/01_original.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

DATA = Path(__file__).resolve().parents[1] / "data"
TARGET = "satisfaction"
SEED = 42

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
orig = pd.read_csv(DATA / "original" / "data.csv")
FEATURES = [c for c in test.columns if c != "id"]
CATEGORICAL = train[FEATURES].select_dtypes(exclude="number").columns.tolist()
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
RATINGS = [c for c in NUMERIC if train[c].max() <= 5]
DIST, DEP, ARR = "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"
y = train[TARGET].astype(int).to_numpy()
y_orig = orig[TARGET].astype(int).to_numpy()


def titulo(texto):
    print(f"\n{'=' * 100}\n{texto}\n{'=' * 100}")


titulo("1. Esquema, balance, ausentes y ceros")
print(f"original: {orig.shape}   train: {train.shape}   test: {test.shape}")
print(f"columnas del original que no están en train: {sorted(set(orig.columns) - set(train.columns))}")
print(f"columnas de train que no están en el original: {sorted(set(train.columns) - set(orig.columns))}")
print(f"dtype del target en el original: {orig[TARGET].dtype}")
print(f"tasa de satisfechos: original {y_orig.mean():.6f} | train {y.mean():.6f}")
print(f"ausentes en el original:\n{orig.isna().sum()[orig.isna().sum() > 0]}")
print(f"{ARR} ausente: original {orig[ARR].isna().mean():.4%} | train {train[ARR].isna().mean():.4%}")
print(f"filas duplicadas en el original (features): {orig.duplicated(FEATURES).sum():,}")

print("\nnuméricas: original frente a train")
print(pd.DataFrame({
    "media_orig": orig[NUMERIC].mean(), "media_train": train[NUMERIC].mean(),
    "desv_orig": orig[NUMERIC].std(), "desv_train": train[NUMERIC].std(),
    "pct_cero_orig": (orig[NUMERIC] == 0).mean() * 100, "pct_cero_train": (train[NUMERIC] == 0).mean() * 100,
    "max_orig": orig[NUMERIC].max(), "max_train": train[NUMERIC].max(),
}).round(3))

print("\ncategóricas: reparto (%) y tasa de satisfacción, original frente a train")
for c in CATEGORICAL:
    print(pd.DataFrame({
        "pct_orig": orig[c].value_counts(normalize=True) * 100,
        "pct_train": train[c].value_counts(normalize=True) * 100,
        "sat_orig": pd.Series(y_orig).groupby(orig[c].to_numpy()).mean(),
        "sat_train": pd.Series(y).groupby(train[c].to_numpy()).mean(),
    }).round(3))

print("\nsatisfacción cuando la valoración es 0: original frente a train")
print(pd.DataFrame({
    "pct_cero_orig": (orig[RATINGS] == 0).mean() * 100,
    "pct_cero_train": (train[RATINGS] == 0).mean() * 100,
    "sat_cero_orig": [y_orig[orig[c] == 0].mean() if (orig[c] == 0).any() else np.nan for c in RATINGS],
    "sat_cero_train": [y[train[c] == 0].mean() if (train[c] == 0).any() else np.nan for c in RATINGS],
}).round(3))

titulo("2. ¿Valores exactos copiados?")
filas = []
for c in [DIST, DEP, ARR, "Age"]:
    vo, vt, vs = set(orig[c].dropna()), set(train[c].dropna()), set(test[c].dropna())
    filas.append({
        "col": c, "unicos_orig": len(vo), "unicos_train": len(vt), "unicos_test": len(vs),
        "train_no_en_orig": len(vt - vo), "test_no_en_orig": len(vs - vo), "orig_no_en_train": len(vo - vt),
        "pct_filas_train_en_orig": train[c].isin(vo).mean() * 100,
        "pct_filas_test_en_orig": test[c].isin(vo).mean() * 100,
    })
print(pd.DataFrame(filas).set_index("col").round(3))

rango = np.arange(orig[DIST].min(), orig[DIST].max() + 1)
print(f"\n{DIST}: el original usa {orig[DIST].nunique():,} de los {len(rango):,} enteros de su rango "
      f"({orig[DIST].nunique() / len(rango):.1%}); train usa {train[DIST].nunique():,}")
huecos = set(rango) - set(orig[DIST])
print(f"filas de train con una distancia que el original NO tiene: {train[DIST].isin(huecos).sum():,} "
      f"({train[DIST].isin(huecos).mean():.4%})")
print("\nlas 10 distancias más frecuentes (filas y % del total):")
print(pd.DataFrame({
    "n_orig": orig[DIST].value_counts().head(10),
    "pct_orig": (orig[DIST].value_counts(normalize=True) * 100).head(10),
    "pct_train": (train[DIST].value_counts(normalize=True) * 100).reindex(orig[DIST].value_counts().head(10).index),
}).round(3))
frec = pd.DataFrame({"orig": orig[DIST].value_counts(normalize=True), "train": train[DIST].value_counts(normalize=True)}).fillna(0)
print(f"correlación entre la frecuencia de cada distancia en el original y en train: {frec.corr().iloc[0, 1]:.4f}")

titulo("3. ¿Llevan target asociado los valores exactos de Flight Distance?")
por_dist = pd.DataFrame({
    "sat_orig": pd.Series(y_orig).groupby(orig[DIST].to_numpy()).mean(),
    "n_orig": orig[DIST].value_counts(),
    "sat_train": pd.Series(y).groupby(train[DIST].to_numpy()).mean(),
    "n_train": train[DIST].value_counts(),
}).dropna()
print(f"distancias comunes: {len(por_dist):,}")
print(f"correlación (sin ponderar) de la tasa por distancia, original frente a train: "
      f"{por_dist[['sat_orig', 'sat_train']].corr().iloc[0, 1]:.4f}")
grandes = por_dist[(por_dist["n_orig"] >= 50) & (por_dist["n_train"] >= 200)]
print(f"lo mismo solo con distancias de >=50 filas en el original y >=200 en train ({len(grandes):,}): "
      f"{grandes[['sat_orig', 'sat_train']].corr().iloc[0, 1]:.4f}")
print(f"desviación típica de la tasa de train entre esas distancias: {grandes['sat_train'].std():.4f}")
print("\nejemplos: distancias vecinas con muchas filas (¿saltos bruscos entre enteros contiguos?)")
top = por_dist.sort_values("n_train", ascending=False).head(15).sort_index()
print(top.round(3))

# AUC de una sola variable, siempre fuera de fold: valor exacto frente a versión suavizada
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, y))


def te_oof(claves, suavizado=20):
    """Target encoding fuera de fold de una clave (Series), con suavizado hacia la media."""
    oof = np.zeros(len(train))
    for tr, va in folds:
        media = y[tr].mean()
        g = pd.DataFrame({"k": claves.iloc[tr].to_numpy(), "y": y[tr]}).groupby("k")["y"].agg(["sum", "count"])
        enc = (g["sum"] + suavizado * media) / (g["count"] + suavizado)
        oof[va] = claves.iloc[va].map(enc).fillna(media).to_numpy()
    return oof


print("\nAUC fuera de fold usando solo Flight Distance:")
print(f"  target encoding del valor exacto ({train[DIST].nunique():,} valores): "
      f"{roc_auc_score(y, te_oof(train[DIST])):.5f}")
for q in [20, 100, 500]:
    tramos = pd.qcut(train[DIST], q, duplicates="drop").cat.codes
    print(f"  target encoding de {tramos.nunique()} tramos por cuantiles:   "
          f"{roc_auc_score(y, te_oof(pd.Series(tramos))):.5f}")

# ¿Sirve la tasa del ORIGINAL por distancia exacta como feature? (no hay fuga: son otros datos)
media_o = y_orig.mean()
g = pd.DataFrame({"k": orig[DIST], "y": y_orig}).groupby("k")["y"].agg(["sum", "count"])
enc_orig = (g["sum"] + 20 * media_o) / (g["count"] + 20)
print(f"  tasa del original por distancia exacta, aplicada a train:  "
      f"{roc_auc_score(y, train[DIST].map(enc_orig).fillna(media_o)):.5f}")

titulo("4. ¿Filas enteras del original repetidas en el sintético?")
orig_u = orig[FEATURES + [TARGET]].drop_duplicates(FEATURES)
en_train = train[FEATURES + [TARGET]].merge(orig_u, on=FEATURES, how="inner", suffixes=("", "_orig"))
en_test = test[FEATURES].merge(orig_u[FEATURES], on=FEATURES, how="inner")
print(f"filas de train idénticas a una del original (21 features): {len(en_train):,}")
if len(en_train):
    print(f"  de ellas, con el mismo target: {(en_train[TARGET] == en_train[TARGET + '_orig']).mean():.4%}")
print(f"filas de test idénticas a una del original (21 features): {len(en_test):,}")

# Coincidencia parcial: solo las 13 valoraciones + 4 categóricas (sin edad, distancia ni retrasos)
PARCIAL = RATINGS + CATEGORICAL
combos_o = orig.groupby(PARCIAL).size().rename("n_orig").reset_index()
tr_parcial = train[PARCIAL].merge(combos_o, on=PARCIAL, how="left")
te_parcial = test[PARCIAL].merge(combos_o, on=PARCIAL, how="left")
print(f"\nfilas cuya combinación de 13 valoraciones + 4 categóricas existe en el original: "
      f"train {tr_parcial['n_orig'].notna().mean():.4%} | test {te_parcial['n_orig'].notna().mean():.4%}")
print(f"combinaciones distintas de esas 17 columnas: original {len(combos_o):,} | "
      f"train {len(train.drop_duplicates(PARCIAL)):,}")

titulo("5. ¿Aprende lo mismo un modelo en el original que en train?")
import lightgbm as lgb

Xo, Xt = orig[FEATURES].copy(), train[FEATURES].copy()
for c in CATEGORICAL:
    niveles = sorted(set(Xo[c]) | set(Xt[c]))
    Xo[c] = pd.Categorical(Xo[c], categories=niveles)
    Xt[c] = pd.Categorical(Xt[c], categories=niveles)
m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
m.fit(Xo, y_orig)
print(f"LightGBM entrenado SOLO con el original (500 árboles), AUC sobre todo train: "
      f"{roc_auc_score(y, m.predict_proba(Xt)[:, 1]):.5f}")
oof_o = np.zeros(len(orig))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xo, y_orig):
    mm = lgb.LGBMClassifier(**m.get_params()).fit(Xo.iloc[tr], y_orig[tr])
    oof_o[va] = mm.predict_proba(Xo.iloc[va])[:, 1]
print(f"el mismo modelo, CV de 5 folds dentro del original: {roc_auc_score(y_orig, oof_o):.5f}")
