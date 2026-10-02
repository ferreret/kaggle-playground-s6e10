"""03 · Gráficas para entender la EDA, la comparación con el original y la baseline.

Son gráficas de trabajo (en español) con el estilo de `plots.py`. Las que sobrevivan pasan
al notebook, traducidas. Las dos últimas necesitan las salidas de `02_baseline.py`.

Uso: uv run python experiments/03_figuras.py
Salidas (ignoradas por git): experiments/outputs/figs/*.png y su texto alternativo en .txt
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from playground_s6e10.plots import DARK_GREY, MID_GREY, annotate, guide_line, save_shareable, shareable_figure

RAIZ = Path(__file__).resolve().parents[1]
DATA, OUT = RAIZ / "data", RAIZ / "experiments" / "outputs"
FIGS = OUT / "figs"
TARGET = "satisfaction"
PIE = "Kaggle Playground S6E10 · gráfica de trabajo (experiments/03_figuras.py)"

# Paleta: dos series categóricas (azul, naranja) y un par divergente azul <-> rojo con gris en medio
AZUL, NARANJA, ROJO, GRIS_MEDIO, REJILLA = "#2a78d6", "#eb6834", "#e34948", "#f0efec", "#e1e0d9"

train = pd.read_csv(DATA / "train.csv")
orig = pd.read_csv(DATA / "original" / "data.csv")
y = train[TARGET].astype(int)
y_orig = orig[TARGET].astype(int)
BASE = y.mean()
FEATURES = [c for c in train.columns if c not in ("id", TARGET)]
NUMERIC = train[FEATURES].select_dtypes(include="number").columns.tolist()
RATINGS = [c for c in NUMERIC if train[c].max() <= 5]
DIST, DEP = "Flight Distance", "Departure Delay in Minutes"


def pct(v, dec=0):
    return f"{v * 100:.{dec}f} %".replace(".", ",")


def miles(n):
    return f"{n:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------------------------------
# 1. Valoraciones: tasa de satisfacción por nota (0..5)
# ---------------------------------------------------------------------------------------------
tasa = pd.DataFrame({c: y.groupby(train[c]).mean() for c in RATINGS}).T.reindex(columns=range(6))
tasa = tasa.sort_values(0, ascending=False, na_position="last")
pct_cero = (train[RATINGS] == 0).mean()

fig, ax = shareable_figure(
    title="Un 0 no es «peor que 1»: en los servicios digitales, el 0 va con pasajeros satisfechos",
    subtitle=f"% de pasajeros satisfechos según la nota que dieron a cada servicio · train, {miles(len(train))} filas · "
             f"media global {pct(BASE, 1)}",
    footer=PIE,
)
cmap = LinearSegmentedColormap.from_list("div", [ROJO, GRIS_MEDIO, AZUL])
norm = TwoSlopeNorm(vmin=0.0, vcenter=BASE, vmax=1.0)
ax.pcolormesh(np.arange(7), np.arange(len(tasa) + 1), np.ma.masked_invalid(tasa.to_numpy()),
              cmap=cmap, norm=norm, edgecolors="white", linewidth=2)
for i, (nombre, fila) in enumerate(tasa.iterrows()):
    for j, v in enumerate(fila):
        if np.isnan(v):
            ax.text(j + 0.5, i + 0.5, "sin datos", ha="center", va="center", fontsize=10, color=MID_GREY)
            continue
        lejos = abs(v - BASE) > 0.27
        ax.text(j + 0.5, i + 0.5, pct(v), ha="center", va="center", fontsize=12,
                color="white" if lejos else DARK_GREY, fontweight="bold" if j == 0 else "normal")
ax.set_xticks(np.arange(6) + 0.5, ["0\n(¿no aplica?)", "1", "2", "3", "4", "5"])
ax.set_yticks(np.arange(len(tasa)) + 0.5,
              [f"{c}  ·  0 en el {pct(pct_cero[c], 1)}" for c in tasa.index])
ax.invert_yaxis()
ax.set_xlabel("Nota que dio el pasajero")
ax.tick_params(length=0)
ax.spines[["left", "bottom"]].set_visible(False)
fig.subplots_adjust(left=0.30, right=0.90)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, BASE, 1])
cb.ax.set_yticklabels(["0 %", f"media\n{pct(BASE)}", "100 %"])
cb.ax.tick_params(labelsize=11, colors=DARK_GREY, length=0)
cb.outline.set_visible(False)
save_shareable(fig, FIGS / "01_valoraciones_por_nota.png", alt_text=(
    "Mapa de calor: 13 servicios en filas y la nota de 0 a 5 en columnas; cada celda es el porcentaje de pasajeros "
    f"satisfechos. Con wifi = 0 el {pct(tasa.loc['Inflight wifi service', 0])} está satisfecho, más que con cualquier "
    "nota de 1 a 4. En asiento, entretenimiento o limpieza el 0 se parece al 1. Las notas 1 a 3 quedan por debajo de "
    "la media en casi todos los servicios y las notas 4 y 5 por encima."))
plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 2. Tipo de viaje x clase
# ---------------------------------------------------------------------------------------------
g = (train.assign(y=y).groupby(["Type of Travel", "Class"])["y"].agg(["mean", "size"])
     .reset_index().sort_values("mean"))
etiquetas = [f"{t} · {c}  ({miles(n)} filas)" for t, c, n in zip(g["Type of Travel"], g["Class"], g["size"])]

fig, ax = shareable_figure(
    title="En viaje personal casi nadie está satisfecho, vaya en la clase que vaya",
    subtitle=f"% de pasajeros satisfechos por tipo de viaje y clase · train, {miles(len(train))} filas",
    footer=PIE,
)
ax.barh(etiquetas, g["mean"], color=AZUL, height=0.6)
for i, v in enumerate(g["mean"]):
    ax.text(v + 0.008, i, pct(v, 1), va="center", fontsize=13, color=DARK_GREY)
ax.set_xlim(0, 0.85)
ax.set_xticks(np.arange(0, 0.81, 0.2), [pct(v) for v in np.arange(0, 0.81, 0.2)])
ax.set_xlabel("Pasajeros satisfechos")
ax.grid(axis="x", color=REJILLA, lw=1)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
guide_line(ax, x=BASE)
ax.text(BASE + 0.008, -0.45, f"media global {pct(BASE, 1)}", color="#d62728", fontsize=11, va="center")
fig.subplots_adjust(left=0.30)
save_shareable(fig, FIGS / "02_viaje_por_clase.png", alt_text=(
    "Barras horizontales con el porcentaje de satisfechos en las seis combinaciones de tipo de viaje y clase. "
    "Las tres de viaje personal están en torno al 10 %. En viaje de negocios: Eco 26 %, Eco Plus 35 % y Business 73 %. "
    f"La media global es {pct(BASE, 1)}."))
plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 3. Distancia: valor exacto frente a tendencia suave
# ---------------------------------------------------------------------------------------------
por_dist = pd.DataFrame({
    "sat_train": y.groupby(train[DIST]).mean(), "n_train": train[DIST].value_counts(),
    "sat_orig": y_orig.groupby(orig[DIST]).mean(), "n_orig": orig[DIST].value_counts(),
})
frecuentes = por_dist[por_dist["n_train"] >= 500]
tramos = pd.qcut(train[DIST], 20, duplicates="drop")
suave = train.assign(y=y).groupby(tramos, observed=True).agg(x=(DIST, "median"), tasa=("y", "mean"))
cubre = frecuentes["n_train"].sum() / len(train)

fig, ax = shareable_figure(
    title="Cada distancia exacta tiene su propia tasa de satisfacción, no la de su tramo",
    subtitle=f"Cada punto es una distancia exacta con 500 filas o más ({miles(len(frecuentes))} distancias, "
             f"{pct(cubre)} de train); el tamaño es el nº de filas · línea: 20 tramos de igual tamaño",
    footer=PIE,
)
ax.scatter(frecuentes.index, frecuentes["sat_train"], s=frecuentes["n_train"] / 25, color=AZUL, alpha=0.45,
           edgecolors="white", linewidths=0.8, label="una distancia exacta")
ax.plot(suave["x"], suave["tasa"], color=DARK_GREY, lw=2, label="tendencia (20 tramos)")
ax.set_xlabel("Distancia del vuelo (millas)")
ax.set_ylabel("Pasajeros satisfechos")
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1.01, 0.2), [pct(v) for v in np.arange(0, 1.01, 0.2)])
ax.grid(axis="y", color=REJILLA, lw=1)
ax.set_axisbelow(True)
for d, dx, dy in [(594, 420, 0.22), (641, 500, -0.14), (2475, -900, 0.17)]:
    if d in frecuentes.index:
        v = frecuentes.loc[d, "sat_train"]
        annotate(ax, f"{d} millas: {pct(v, 1)}", xy=(d, v), xytext=(d + dx, v + dy))
ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=DARK_GREY)
save_shareable(fig, FIGS / "03_distancia_exacta.png", alt_text=(
    "Dispersión: distancia del vuelo en el eje horizontal y porcentaje de satisfechos en el vertical; un punto por "
    "cada distancia exacta con 500 filas o más. Una línea con la tendencia por tramos sube con la distancia, pero los "
    "puntos se dispersan mucho alrededor: 594 millas tiene un 36,2 % de satisfechos y 641 millas un 22,6 %."))
plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 4. Tasa por distancia: original frente a sintético
# ---------------------------------------------------------------------------------------------
comunes = por_dist.dropna()
grandes = comunes[(comunes["n_orig"] >= 50) & (comunes["n_train"] >= 200)]
corr = grandes[["sat_orig", "sat_train"]].corr().iloc[0, 1]

fig, ax = shareable_figure(
    title="El sintético hereda del original la tasa de satisfacción de cada distancia",
    subtitle=f"Un punto por distancia exacta con ≥50 filas en el original y ≥200 en train ({miles(len(grandes))} "
             f"distancias) · correlación {corr:.2f}".replace(".", ","),
    footer=PIE,
)
ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=MID_GREY)
ax.scatter(grandes["sat_orig"], grandes["sat_train"], s=grandes["n_train"] / 25, color=AZUL, alpha=0.45,
           edgecolors="white", linewidths=0.8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ticks = np.arange(0, 1.01, 0.2)
ax.set_xticks(ticks, [pct(v) for v in ticks])
ax.set_yticks(ticks, [pct(v) for v in ticks])
ax.set_xlabel("Satisfechos con esa distancia en el dataset original (real)")
ax.set_ylabel("Satisfechos con esa distancia en train (sintético)")
ax.grid(color=REJILLA, lw=1)
ax.set_axisbelow(True)
ax.set_aspect("equal", adjustable="box")
ax.text(0.83, 0.78, "misma tasa\nen los dos", color=MID_GREY, fontsize=11, ha="left", va="top")
save_shareable(fig, FIGS / "04_distancia_original_vs_train.png", alt_text=(
    "Dispersión: en el eje horizontal el porcentaje de satisfechos de cada distancia exacta en el dataset original y en "
    f"el vertical el mismo porcentaje en train. Los {len(grandes)} puntos siguen la diagonal con una correlación de "
    f"{corr:.2f}: las distancias con muchos satisfechos en el original también los tienen en el sintético."))
plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 5. Retrasos de salida: original frente a sintético
# ---------------------------------------------------------------------------------------------
cortes, nombres = [-1, 0, 15, 60, 180, 1e6], ["sin retraso", "1-15 min", "16-60 min", "1-3 h", "más de 3 h"]
r_orig = pd.cut(orig[DEP], cortes, labels=nombres).value_counts(normalize=True).reindex(nombres)
r_train = pd.cut(train[DEP], cortes, labels=nombres).value_counts(normalize=True).reindex(nombres)

fig, ax = shareable_figure(
    title="El generador borró casi todos los retrasos: de un 44 % de vuelos con retraso a un 9 %",
    subtitle=f"% de vuelos por tramo de retraso de salida · original {miles(len(orig))} filas (media "
             f"{orig[DEP].mean():.1f} min) frente a train {miles(len(train))} filas (media {train[DEP].mean():.1f} min)"
             .replace(".1 ", ",1 ").replace("14.7", "14,7").replace("1.2", "1,2"),
    footer=PIE,
)
x, ancho = np.arange(len(nombres)), 0.38
for dx, serie, color, nombre in [(-ancho / 2 - 0.01, r_orig, AZUL, "original (real)"),
                                 (ancho / 2 + 0.01, r_train, NARANJA, "train (sintético)")]:
    ax.bar(x + dx, serie, ancho, color=color, label=nombre)
    for xi, v in zip(x + dx, serie):
        ax.text(xi, v + 0.012, pct(v, 1), ha="center", fontsize=12, color=DARK_GREY)
ax.set_xticks(x, nombres)
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1.01, 0.2), [pct(v) for v in np.arange(0, 1.01, 0.2)])
ax.set_ylabel("Vuelos")
ax.set_xlabel("Retraso de salida")
ax.grid(axis="y", color=REJILLA, lw=1)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
ax.legend(loc="upper right", frameon=False, fontsize=13, labelcolor=DARK_GREY)
save_shareable(fig, FIGS / "05_retrasos_original_vs_train.png", alt_text=(
    "Barras agrupadas con el porcentaje de vuelos en cinco tramos de retraso de salida, original frente a train. "
    f"Sin retraso: {pct(r_orig.iloc[0], 1)} en el original y {pct(r_train.iloc[0], 1)} en train. En todos los tramos "
    "con retraso, train tiene muchos menos vuelos que el original."))
plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 6 y 7. Baseline (si ya se ha ejecutado 02_baseline.py)
# ---------------------------------------------------------------------------------------------
if (OUT / "02_oof_lgbm_raw.npy").exists():
    import json

    from sklearn.metrics import roc_auc_score

    oof = np.load(OUT / "02_oof_lgbm_raw.npy")
    res = json.loads((OUT / "02_baseline.json").read_text())
    cv = f"{res['cv_mean']:.5f} ± {res['cv_std']:.5f}".replace(".", ",")
    assert abs(roc_auc_score(y, oof) - res["oof"]) < 1e-9

    yy = y.to_numpy()
    cortes_p, bandas = [0, 0.1, 0.3, 0.7, 0.9, 1.0], ["0-10 %", "10-30 %", "30-70 %", "70-90 %", "90-100 %"]
    banda = pd.cut(oof, cortes_p, labels=bandas, include_lowest=True)
    tabla = pd.crosstab(banda, yy).reindex(bandas) / len(oof)  # columnas 0 (no satisfecho) y 1 (satisfecho)
    errores = (oof > 0.5) != yy
    seguros_mal = ((oof > 0.9) & (yy == 0)) | ((oof < 0.1) & (yy == 1))
    print("\npredicción fuera de fold por banda (% de train):")
    print((tabla * 100).round(2).rename(columns={0: "no satisfechos", 1: "satisfechos"}))
    print(f"errores con corte en 0,5: {errores.sum():,} ({errores.mean():.2%}); de ellos, con el modelo a más del "
          f"90 % en el lado contrario: {seguros_mal.sum():,} ({seguros_mal.sum() / errores.sum():.1%})")

    fig, ax = shareable_figure(
        title="Casi la mitad de los errores son casos «seguros» con la etiqueta contraria",
        subtitle=f"Reparto de train según la probabilidad que predice el modelo sin haber visto al pasajero (fuera de "
                 f"fold) · LightGBM, 5 folds, AUC {cv}",
        footer=PIE,
    )
    x = np.arange(len(bandas))
    ax.bar(x, tabla[0], 0.62, color=NARANJA, label="no satisfechos (etiqueta real)", edgecolor="white", linewidth=2)
    ax.bar(x, tabla[1], 0.62, bottom=tabla[0], color=AZUL, label="satisfechos (etiqueta real)", edgecolor="white",
           linewidth=2)
    for xi, (ns, s) in enumerate(zip(tabla[0], tabla[1])):
        ax.text(xi, ns + s + 0.008, pct(ns + s, 1), ha="center", fontsize=13, color=DARK_GREY, fontweight="bold")
    annotate(ax, f"{pct(tabla.loc['0-10 %', 1], 1)} de train: satisfechos a los que\nel modelo da menos de un 10 %",
             xy=(0.0, tabla.loc["0-10 %"].sum() - tabla.loc["0-10 %", 1] / 2), xytext=(0.55, 0.40))
    annotate(ax, f"{pct(tabla.loc['90-100 %', 0], 1)} de train: no satisfechos a los que\nel modelo da más de un 90 %",
             xy=(4.0, tabla.loc["90-100 %", 0] / 2), xytext=(2.25, 0.20))
    ax.set_xticks(x, bandas)
    ax.set_ylim(0, 0.55)
    ax.set_yticks(np.arange(0, 0.51, 0.1), [pct(v) for v in np.arange(0, 0.51, 0.1)])
    ax.set_xlabel("Probabilidad de «satisfecho» que predice el modelo")
    ax.set_ylabel("Pasajeros de train")
    ax.grid(axis="y", color=REJILLA, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", frameon=False, fontsize=13, labelcolor=DARK_GREY, ncols=2)
    save_shareable(fig, FIGS / "06_baseline_predicciones.png", alt_text=(
        "Barras apiladas: cinco bandas de probabilidad predicha en el eje horizontal y porcentaje de pasajeros de train "
        f"en el vertical, partido entre satisfechos y no satisfechos. El {pct(tabla.loc['0-10 %'].sum(), 1)} de train "
        f"recibe menos de un 10 % y el {pct(tabla.loc['90-100 %'].sum(), 1)} más de un 90 %. En cada extremo hay cerca "
        "de un 1,8 % de train con la etiqueta contraria a la que el modelo da por segura."))
    plt.close(fig)

    gan = pd.read_csv(OUT / "02_gain_lgbm_raw.csv", index_col=0)["gain"]
    gan = (gan / gan.sum()).sort_values()
    fig, ax = shareable_figure(
        title="En qué se apoya la baseline: reparto de la ganancia entre las 21 variables",
        subtitle="% de la mejora total que aporta cada variable al partir los árboles (ganancia), media de 5 folds · "
                 "LightGBM con las variables tal cual",
        footer=PIE,
    )
    ax.barh(gan.index, gan, color=AZUL, height=0.65)
    for i, v in enumerate(gan):
        ax.text(v + 0.003, i, pct(v, 1), va="center", fontsize=11, color=DARK_GREY)
    ax.set_xlim(0, gan.max() * 1.12)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    fig.subplots_adjust(left=0.24)
    save_shareable(fig, FIGS / "07_baseline_importancia.png", alt_text=(
        "Barras horizontales con el porcentaje de la ganancia total de LightGBM que corresponde a cada una de las 21 "
        f"variables, ordenadas de mayor a menor. La primera es {gan.index[-1]} con el {pct(gan.iloc[-1], 1)}."))
    plt.close(fig)

print("figuras:", *sorted(p.name for p in FIGS.glob("*.png")), sep="\n  ")
