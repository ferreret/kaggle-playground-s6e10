"""00 · EDA mínima del día 1 (S6E10, Predicting Airline Satisfaction).

Preguntas: tamaño, tipos, balance del target, ausentes (Arrival Delay), ceros en las
valoraciones de servicio, y train frente a test.

Uso: uv run python experiments/00_eda.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

DATA = Path(__file__).resolve().parents[1] / "data"
TARGET = "satisfaction"

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
FEATURES = [c for c in test.columns if c != "id"]
CATEGORICAL = train[FEATURES].select_dtypes(exclude="number").columns.tolist()
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
RATINGS = [c for c in NUMERIC if train[c].max() <= 5]
y = train[TARGET].astype(int)


def titulo(texto):
    print(f"\n{'=' * 100}\n{texto}\n{'=' * 100}")


titulo("1. Tamaño y tipos")
print(f"train: {train.shape}   test: {test.shape}")
print(f"ids train: {train['id'].min()}..{train['id'].max()} (únicos: {train['id'].is_unique})")
print(f"ids test:  {test['id'].min()}..{test['id'].max()} (únicos: {test['id'].is_unique})")
print(f"dtype del target: {train[TARGET].dtype}")
print(f"categóricas ({len(CATEGORICAL)}): {CATEGORICAL}")
print(f"valoraciones ({len(RATINGS)}): {RATINGS}")
print(pd.DataFrame({
    "dtype": train[FEATURES].dtypes.astype(str),
    "n_unique_train": train[FEATURES].nunique(),
    "n_unique_test": test[FEATURES].nunique(),
    "min": [train[c].min() for c in FEATURES],
    "max": [train[c].max() for c in FEATURES],
}))

titulo("2. Balance del target")
print(train[TARGET].value_counts().to_frame("n").assign(pct=lambda d: d["n"] / len(train)))
print(f"tasa de satisfechos: {y.mean():.6f}")

titulo("3. Ausentes")
nulos = pd.DataFrame({
    "train_n": train[FEATURES].isna().sum(),
    "train_pct": train[FEATURES].isna().mean() * 100,
    "test_n": test[FEATURES].isna().sum(),
    "test_pct": test[FEATURES].isna().mean() * 100,
})
print(nulos[nulos[["train_n", "test_n"]].sum(axis=1) > 0])
ARR, DEP = "Arrival Delay in Minutes", "Departure Delay in Minutes"
falta = train[ARR].isna()
print(f"\nsatisfacción con {ARR} ausente:  {y[falta].mean():.4f} (n={falta.sum():,})")
print(f"satisfacción con {ARR} presente: {y[~falta].mean():.4f} (n={(~falta).sum():,})")
print(f"\n{DEP} cuando falta {ARR}:")
print(train.loc[falta, DEP].describe())
print(f"\n{DEP} cuando NO falta {ARR}:")
print(train.loc[~falta, DEP].describe())
print(f"\ncorrelación Pearson salida/llegada (train, sin nulos): {train[[DEP, ARR]].corr().iloc[0, 1]:.4f}")
print(f"filas con retraso de llegada no entero: {(train[ARR].dropna() % 1 != 0).sum()}")

titulo("4. Ceros en las valoraciones de servicio (escala nominal 1-5; el 0 queda fuera)")
ceros = pd.DataFrame({
    "train_n": (train[RATINGS] == 0).sum(),
    "train_pct": (train[RATINGS] == 0).mean() * 100,
    "test_pct": (test[RATINGS] == 0).mean() * 100,
    "sat_si_cero": [y[train[c] == 0].mean() if (train[c] == 0).any() else np.nan for c in RATINGS],
    "sat_si_no_cero": [y[train[c] != 0].mean() for c in RATINGS],
})
print(ceros.round(4))
n_ceros = (train[RATINGS] == 0).sum(axis=1)
print(f"\nfilas con al menos un cero: train {(n_ceros > 0).mean():.4%} | test {((test[RATINGS] == 0).sum(axis=1) > 0).mean():.4%}")
print("\nsatisfacción según el número de ceros en la fila:")
print(y.groupby(n_ceros).agg(["mean", "size"]))

titulo("4b. Satisfacción por valor de cada valoración (filas = valoración, columnas = 0..5)")
print(pd.DataFrame({c: y.groupby(train[c]).mean() for c in RATINGS}).T.round(3))
print("\nreparto de valores (% de filas, train):")
print(pd.DataFrame({c: train[c].value_counts(normalize=True).sort_index() * 100 for c in RATINGS}).T.round(2))

titulo("5. Satisfacción por categoría")
for c in CATEGORICAL:
    print(f"\n{c}")
    print(pd.DataFrame({
        "train_pct": train[c].value_counts(normalize=True) * 100,
        "test_pct": test[c].value_counts(normalize=True) * 100,
        "sat": y.groupby(train[c]).mean(),
    }).round(4))
print("\nType of Travel x Class (tasa de satisfacción y nº de filas):")
print(pd.pivot_table(train.assign(y=y), index="Type of Travel", columns="Class", values="y", aggfunc=["mean", "size"]).round(4))

titulo("6. Numéricas continuas")
CONT = [c for c in NUMERIC if c not in RATINGS]
print(pd.concat({"train": train[CONT].describe().T, "test": test[CONT].describe().T}, axis=1).round(2))
for c in [DEP, ARR]:
    print(f"{c}: % de ceros train {(train[c] == 0).mean():.4%} | test {(test[c] == 0).mean():.4%}")
print("\nsatisfacción por tramo de edad:")
print(y.groupby(pd.cut(train["Age"], [0, 17, 24, 39, 60, 120])).agg(["mean", "size"]).round(4))
print("\nsatisfacción por tramo de distancia:")
print(y.groupby(pd.qcut(train["Flight Distance"], 5, duplicates="drop")).agg(["mean", "size"]).round(4))
print("\nsatisfacción por tramo de retraso de salida:")
print(y.groupby(pd.cut(train[DEP], [-1, 0, 15, 60, 180, 1e6])).agg(["mean", "size"]).round(4))

titulo("7. Train frente a test")
filas = []
for c in NUMERIC:
    a, b = train[c].dropna(), test[c].dropna()
    filas.append({
        "col": c, "media_train": a.mean(), "media_test": b.mean(),
        "dif_en_desv": (a.mean() - b.mean()) / a.std(),
        "solo_en_test": len(set(b.unique()) - set(a.unique())),
    })
print(pd.DataFrame(filas).set_index("col").round(4))

# Validación adversaria: ¿un modelo distingue train de test?
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ambos = pd.concat([train[FEATURES], test[FEATURES]], ignore_index=True)
for c in CATEGORICAL:
    ambos[c] = ambos[c].astype("category")
es_test = np.r_[np.zeros(len(train)), np.ones(len(test))]
oof = np.zeros(len(ambos))
for tr, va in StratifiedKFold(3, shuffle=True, random_state=42).split(ambos, es_test):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42)
    m.fit(ambos.iloc[tr], es_test[tr])
    oof[va] = m.predict_proba(ambos.iloc[va])[:, 1]
print(f"\nAUC de la validación adversaria (0,5 = indistinguibles): {roc_auc_score(es_test, oof):.4f}")

titulo("8. Filas duplicadas")
print(f"duplicadas dentro de train (features): {train.duplicated(FEATURES).sum():,}")
print(f"duplicadas dentro de train (features + target): {train.duplicated(FEATURES + [TARGET]).sum():,}")
grupos = train.groupby(FEATURES, dropna=False)[TARGET].agg(["nunique", "size"])
print(f"combinaciones de features con las dos etiquetas: {(grupos['nunique'] > 1).sum():,}")
comunes = test[FEATURES].merge(train[FEATURES].drop_duplicates(), on=FEATURES, how="inner")
print(f"filas de test cuya combinación exacta de features aparece en train: {len(comunes):,}")
