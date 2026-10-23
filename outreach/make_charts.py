"""Gráficas para el hilo de X, a 1600x900 exactos y con el estilo del notebook.

Las dos primeras salen del código del notebook publicado (celdas 2, 4, 6, 8, 14 y 39), ejecutado tal cual,
sin el recorte de `bbox_inches="tight"` que aplica Jupyter al mostrarlas. La tercera es nueva: resume la tabla de
la Part 9.4 y las ganancias de la Part 7.4, con cifras medidas (CV del notebook y LB público de Kaggle).

Uso, desde la raíz del repo:
    uv run python outreach/make_charts.py
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outreach" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

cells = json.loads((ROOT / "notebooks" / "s6e10-step-by-step.ipynb").read_text())["cells"]
src = {i: "".join(cells[i]["source"]) for i in (2, 4, 6, 8, 14, 39)}

os.chdir(ROOT / "notebooks")  # el notebook lee los datos de ../data
ns = {}
for i in (2, 4, 6, 8):
    exec(src[i].replace("train.head()", "").replace("pd.Series(y).map", "_ = pd.Series(y).map"), ns)
exec('y_orig = orig[TARGET].astype(int).to_numpy()', ns)  # de la celda 35


def run_and_save(cell, name):
    ns["plt"].show = lambda: plt.gcf().savefig(OUT / name, dpi=100)
    exec(src[cell], ns)
    plt.close("all")


run_and_save(14, "01_zero_rating_heatmap.png")
run_and_save(39, "02_exact_distance.png")

# Cifras para el texto alternativo
rate, zero_share = ns["rate"], ns["zero_share"]
print("tasa de satisfechos por valoración (filas ordenadas como en la gráfica):")
print((rate * 100).round(0).to_string())
print("\n% de ceros:", (zero_share * 100).round(1).sort_values(ascending=False).head(4).to_dict())
frequent, smooth = ns["frequent"], ns["smooth"]
print(f"\ndistancias con 500+ filas: {len(frequent)} · {frequent['n_train'].sum() / len(ns['train']):.0%} de train")
print("anotadas:", {d: round(frequent.loc[d, 'rate_train'] * 100, 1) for d in (594, 641, 2475)})
print("tasa en las 20 franjas: de", round(smooth["rate"].min() * 100, 1), "a", round(smooth["rate"].max() * 100, 1))
print("rango de tasas de las distancias:", round(frequent["rate_train"].min() * 100, 1), "-",
      round(frequent["rate_train"].max() * 100, 1))

# --- 03: lo que prometió la CV frente a lo que dio el LB público ----------------------------------------------
# Ganancia de cada cambio sobre el modelo anterior, en unidades de 0,0001 de AUC.
# Fuentes: tabla de la Part 7.4 (TE sobre LightGBM sin más) y de la Part 9.4 (el resto, sobre el modelo con TE).
changes = [
    ("Exact-distance target encoding\n(vs raw-feature LightGBM)", 12.7, 12.5),
    ("Teacher feature + tuned parameters", 5.2, 0.0),
    ("Teacher feature", 2.7, -1.3),
    ("Tuned parameters", 2.4, -1.6),
    ("Control: same model, seed 43", -0.9, 1.2),
]
shareable_figure, annotate = ns["shareable_figure"], ns["annotate"]
BLUE, ORANGE, DARK_GREY, MID_GREY, GRID = ns["BLUE"], ns["ORANGE"], ns["DARK_GREY"], ns["MID_GREY"], ns["GRID"]

fig, ax = shareable_figure(
    title="Four changes improved the CV; the public leaderboard confirmed only one",
    subtitle="ROC AUC gain of each change over the model before it, in units of 0.0001 · 5-fold CV vs public "
             "leaderboard · LightGBM",
)
h = 0.34
for i, (label, cv, lb) in enumerate(changes):
    for dy, v, color in [(-h / 2, cv, BLUE), (h / 2, lb, ORANGE)]:
        ax.barh(i + dy, v, height=h - 0.04, color=color, edgecolor="white", linewidth=2)
        ax.text(v + (0.25 if v >= 0 else -0.25), i + dy, f"{v:+.1f}".replace("-", "−"), va="center",
                ha="left" if v >= 0 else "right", fontsize=12, color=DARK_GREY)
handles = [Patch(color=BLUE, label="CV gain (mean of 5 folds)"), Patch(color=ORANGE, label="Public leaderboard gain")]
ax.axvspan(-1.2, 1.2, color=GRID, alpha=0.6, zorder=0)
ax.axvline(0, color=MID_GREY, lw=1)
ax.text(0, 4.62, "what a new seed alone moves (up to 1.2)", ha="center", va="bottom", fontsize=11, color=MID_GREY)
ax.set_yticks(range(len(changes)), [c[0] for c in changes])
ax.invert_yaxis()
ax.set_xlim(-4, 15)
ax.set_ylim(4.8, -0.6)
ax.set_xlabel("AUC gain (× 0.0001)")
ax.tick_params(axis="y", length=0)
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=13, labelcolor=DARK_GREY)
annotate(ax, "CV +0.00127, leaderboard +0.00125:\nthe one gain both sides agree on", xy=(12.7, -0.17),
         xytext=(7.4, 0.95))
annotate(ax, "+0.00052 on every fold,\n+0.00000 on the leaderboard", xy=(5.2, 0.83), xytext=(7.4, 2.05))
fig.subplots_adjust(left=0.27)
fig.savefig(OUT / "03_cv_vs_leaderboard.png", dpi=100)
plt.close(fig)

for p in sorted(OUT.glob("*.png")):
    print(p.name, plt.imread(p).shape[1::-1])
