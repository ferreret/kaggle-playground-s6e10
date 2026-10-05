"""08 · Gráficas para entender la red neuronal del experimento 07.

Gráficas de trabajo (en español) con el estilo de `plots.py`. Necesita las salidas de `07_red_neuronal.py`.

Uso: uv run python experiments/08_figuras_red.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from playground_s6e10.cv import OUT
from playground_s6e10.plots import DARK_GREY, MID_GREY, save_shareable, shareable_figure

FIGS = OUT / "figs"
PIE = "Kaggle Playground S6E10 · gráfica de trabajo (experiments/08_figuras_red.py)"
AZUL, NARANJA, REJILLA, FONDO = "#2a78d6", "#eb6834", "#e1e0d9", "#eef4fc"
res = json.loads((OUT / "07_red_neuronal.json").read_text())
REF = float(np.mean(json.loads((OUT / "04_distancia_te.json").read_text())["te"]["fold_scores"]))
coma = lambda v, d=5: f"{v:.{d}f}".replace(".", ",")

# ---------------------------------------------------------------------------------------------
# 1. Esquema de la red
# ---------------------------------------------------------------------------------------------
fig, ax = shareable_figure(
    title="La red: cada dato del pasajero se convierte en números, y tres capas los combinan",
    subtitle="Perceptrón multicapa con embeddings · 89 números de entrada · 170.365 parámetros que se aprenden",
    footer=PIE,
)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.08)


def caja(x, y, w, h, titulo, texto, color=FONDO):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec=AZUL, lw=1.5))
    ax.text(x + w / 2, y + h - 0.28, titulo, ha="center", va="top", fontsize=12.5, fontweight="bold", color=DARK_GREY)
    ax.text(x + w / 2, y + h - 0.78, texto, ha="center", va="top", fontsize=11, color=DARK_GREY, linespacing=1.35)


def flecha(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": MID_GREY, "lw": 1.6})


caja(0.2, 6.3, 3.5, 2.2, "13 valoraciones y 4 categorías", "cada valor -> su fila de\n4 números (embedding)\n17 × 4 = 68 números")
caja(0.2, 3.5, 3.5, 2.2, "Distancia exacta", "cada distancia -> su fila de\n16 números (embedding)\n16 números")
caja(0.2, 0.7, 3.5, 2.2, "Edad, distancia, retrasos", "números reescalados\n+ marca de «falta el dato»\n5 números")
caja(4.7, 3.3, 2.0, 2.6, "Se juntan", "68 + 16 + 5\n= 89 números\npor pasajero", color="white")
for yy in (7.4, 4.6, 1.8):
    flecha(3.75, yy, 4.65, 4.6)
for i, (x, n) in enumerate([(7.6, 256), (9.9, 256), (12.2, 128)]):
    caja(x, 3.0, 1.7, 3.2, f"Capa {i + 1}", f"{n} neuronas\n\ncada una:\nsuma con pesos\n+ «si es negativo,\npon un 0»")
    flecha(x - 0.55 if i else 6.75, 3.6, x - 0.05, 3.6)
caja(14.3, 3.5, 1.5, 2.2, "Salida", "1 número\n-> probabilidad\nde «satisfecho»", color="white")
flecha(13.95, 3.9, 14.25, 3.9)
ax.text(8.0, 1.6, "Al entrenar, la red ajusta a la vez los pesos de las capas y las filas de los embeddings,\n"
                  "para que la probabilidad de salida se parezca a la respuesta real de cada pasajero.",
        fontsize=12, color=MID_GREY, va="top")
save_shareable(fig, FIGS / "08_red_esquema.png", alt_text=(
    "Esquema de izquierda a derecha. Tres grupos de entrada: 13 valoraciones y 4 categorías convertidas en 68 números, "
    "la distancia exacta convertida en 16 números y 5 números reescalados. Se juntan en 89 números, pasan por tres "
    "capas de 256, 256 y 128 neuronas y terminan en un número que se convierte en probabilidad de satisfecho."))
plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 2. Curva de aprendizaje (fold 0, red con distancia)
# ---------------------------------------------------------------------------------------------
h = res["con_distancia"]["historias"][0]
epocas = [e["epoca"] for e in h]
auc = [e["auc_validacion"] for e in h]
perdida = [e["perdida_entrenamiento"] for e in h]
mejor = int(np.argmax(auc))

fig, axes = shareable_figure(
    title="La red sigue mejorando en lo que ha visto, pero deja de mejorar en lo que no ha visto",
    subtitle=f"Red con embedding de la distancia, fold 0 · una época = una pasada por los 559.708 pasajeros de entrenamiento",
    footer=PIE, ncols=2,
)
axes[0].plot(epocas, perdida, color=NARANJA, lw=2, marker="o", ms=7)
axes[0].set_title("Error sobre los pasajeros de entrenamiento (baja = aprende)", loc="left", fontsize=13, color=DARK_GREY)
axes[0].set_ylabel("Error medio (log-loss)")
axes[1].plot(epocas, auc, color=AZUL, lw=2, marker="o", ms=7)
axes[1].set_title("AUC sobre los pasajeros de validación (sube = generaliza)", loc="left", fontsize=13, color=DARK_GREY)
axes[1].set_ylabel("AUC de validación")
axes[1].axvline(epocas[mejor], ls="--", lw=1.2, color="#d62728")
axes[1].annotate(f"mejor época: {epocas[mejor]}\nAUC {coma(auc[mejor])}\nnos quedamos con esta", xy=(epocas[mejor], auc[mejor]),
                 xytext=(epocas[mejor] + 0.6, min(auc) + (max(auc) - min(auc)) * 0.35), fontsize=12, color="#d62728",
                 arrowprops={"arrowstyle": "-", "color": "#d62728", "lw": 1.2})
for a in axes:
    a.set_xlabel("Época")
    a.set_xticks(epocas)
    a.grid(axis="y", color=REJILLA, lw=1)
    a.set_axisbelow(True)
fig.subplots_adjust(top=0.80, wspace=0.22)
save_shareable(fig, FIGS / "08_red_aprendizaje.png", alt_text=(
    f"Dos paneles por época de entrenamiento. A la izquierda, el error sobre los datos de entrenamiento baja en todas "
    f"las {len(epocas)} épocas. A la derecha, el AUC de validación sube hasta la época {epocas[mejor]} "
    f"({coma(auc[mejor])}) y después deja de mejorar."))
plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 3. Mezcla con LightGBM
# ---------------------------------------------------------------------------------------------
fig, ax = shareable_figure(
    title="La red sola pierde contra LightGBM, pero una parte pequeña de red mejora la mezcla",
    subtitle=f"CV (5 folds) de la mezcla según el peso que se da a la red · peso 0 = solo LightGBM ({coma(REF)}), peso 1 = solo la red",
    footer=PIE,
)
for clave, color, nombre in [("con_distancia", AZUL, "red con embedding de la distancia"),
                             ("sin_distancia", NARANJA, "red sin la distancia exacta")]:
    m = res[clave]["mezcla"]
    ws = sorted(float(w) for w in m)
    cv = [m[str(w)]["cv"] for w in ws]
    ax.plot(ws, cv, color=color, lw=2, marker="o", ms=7, label=nombre)
    ax.text(1.015, cv[-1], coma(cv[-1]), color=DARK_GREY, fontsize=11, va="center")
    if clave == "con_distancia":
        i = int(np.argmax(cv))
        ax.annotate(f"mejor mezcla: {int(ws[i] * 100)} % red\n{coma(cv[i])} ({m[str(ws[i])]['dif']:+.5f})".replace(".", ","),
                    xy=(ws[i], cv[i]), xytext=(ws[i] + 0.2, cv[i] + 0.00022), fontsize=12, color="#d62728",
                    arrowprops={"arrowstyle": "-", "color": "#d62728", "lw": 1.2})
ax.set_ylim(top=0.9608)
ax.axhline(REF, ls="--", lw=1.2, color=MID_GREY)
ax.text(0.99, REF - 0.00006, "solo LightGBM", color=MID_GREY, fontsize=11, ha="right", va="top")
ax.set_xlabel("Peso de la red en la mezcla")
ax.set_ylabel("AUC (CV, 5 folds)")
ax.set_xticks(np.arange(0, 1.01, 0.1), [f"{int(w * 100)} %" for w in np.arange(0, 1.01, 0.1)])
ax.yaxis.set_major_formatter(lambda v, _: coma(v, 4))
ax.grid(axis="y", color=REJILLA, lw=1)
ax.set_axisbelow(True)
ax.legend(loc="lower left", frameon=False, fontsize=13, labelcolor=DARK_GREY)
save_shareable(fig, FIGS / "08_red_mezcla.png", alt_text=(
    "Dos curvas del AUC de la mezcla según el peso de la red, de 0 a 100 %. Las dos empiezan en el AUC de LightGBM solo "
    f"({coma(REF)}), suben un poco con un peso del 20 al 30 % y después bajan hasta el AUC de la red sola. La red con "
    "embedding de la distancia queda por encima de la otra en todo el recorrido."))
plt.close(fig)
print("figuras:", *sorted(p.name for p in FIGS.glob("08_*.png")), sep="\n  ")

# ---------------------------------------------------------------------------------------------
# 4. De 21 datos a 89 números: un pasajero real (necesita 07_ejemplo_entrada.npz)
# ---------------------------------------------------------------------------------------------
import pandas as pd
from matplotlib.patches import Rectangle

from playground_s6e10.cv import DATA

ej = np.load(OUT / "07_ejemplo_entrada.npz", allow_pickle=True)
fila = int(ej["fila"])
pasajero = pd.read_csv(DATA / "train.csv", skiprows=range(1, fila + 1), nrows=1).iloc[0]
VERDE, GRIS = "#1baf7a", "#898781"
SUAVE = {AZUL: "#dbe9fa", VERDE: "#d9f2e8", NARANJA: "#fde3d8", GRIS: "#ececea"}
N_VAL = 13

fig, ax = shareable_figure(
    title="De 21 datos a 89 números: así entra un pasajero real en la red",
    subtitle=f"Cada casilla es un número que recibe la primera capa · pasajero id {int(pasajero['id'])} de train · "
             f"red con embedding de la distancia, fold 0",
    footer=PIE,
)
ax.set_xlim(0, 16.6)
ax.set_ylim(26.2, -1.2)
ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.05)
X0, ANCHO = 4.75, 0.62


def casillas(y, valores, color, x0=X0, resaltar=False):
    for j, v in enumerate(valores):
        ax.add_patch(Rectangle((x0 + j * ANCHO, y - 0.42), ANCHO - 0.05, 0.84, fc=SUAVE[color], ec=color,
                               lw=2.2 if resaltar else 0.9))
        ax.text(x0 + j * ANCHO + (ANCHO - 0.05) / 2, y, f"{v:+.1f}".replace(".", ","), ha="center", va="center",
                fontsize=8.5, color=DARK_GREY)


def etiqueta(y, nombre, valor):
    ax.text(X0 - 0.15, y, f"{nombre} = {valor}", ha="right", va="center", fontsize=10, color=DARK_GREY)


def grupo(y, texto, color):
    ax.text(0.1, y, texto, ha="left", va="center", fontsize=11.5, color=color, fontweight="bold")


y = 0
grupo(y, "13 valoraciones  ->  cada una, la fila de 4 números de su nota  ->  13 × 4 = 52 números", AZUL)
for j, col in enumerate(ej["columnas"][:N_VAL]):
    y += 1
    etiqueta(y, col, int(pasajero[col]))
    casillas(y, ej["embeddings"][j], AZUL, resaltar=(j == 0))
y += 1.5
grupo(y, "4 categorías  ->  cada una, la fila de 4 números de su valor  ->  4 × 4 = 16 números", VERDE)
for j, col in enumerate(ej["columnas"][N_VAL:], start=N_VAL):
    y += 1
    etiqueta(y, col, pasajero[col])
    casillas(y, ej["embeddings"][j], VERDE)
y += 1.5
grupo(y, "La distancia como «ruta»  ->  la fila de 16 números de esa distancia exacta  ->  16 números", NARANJA)
y += 1
etiqueta(y, "Flight Distance", int(pasajero["Flight Distance"]))
casillas(y, ej["distancia"], NARANJA)
y += 1.5
grupo(y, "5 números tal cual, reescalados  ->  5 números", GRIS)
y += 1
nombres = ["edad", "distancia", "retraso salida", "retraso llegada", "¿falta llegada?"]
for j, (n, v) in enumerate(zip(nombres, ej["numericas"])):
    x = X0 + j * 2.25
    casillas(y, [v], GRIS, x0=x)
    ax.text(x + ANCHO + 0.05, y, n, ha="left", va="center", fontsize=9.5, color=DARK_GREY)
ax.text(X0 - 0.15, y, "números", ha="right", va="center", fontsize=10, color=DARK_GREY)
ax.text(X0, y + 1.5, "Total:  52 + 16 + 16 + 5  =  89 números entran en la primera capa", ha="left", va="center",
        fontsize=14, color=DARK_GREY, fontweight="bold")

# Detalle: la tabla del wifi, con la fila de la nota de este pasajero resaltada
TX, TY, nota = 10.4, 2.2, int(pasajero[ej["columnas"][0]])
ax.text(TX - 0.9, TY - 1.5, "De dónde salen esos 4 números: la tabla del wifi", fontsize=12, color=AZUL, fontweight="bold")
ax.text(TX - 0.9, TY - 0.7, "Una fila por nota. La red aprendió estos números al entrenar.", fontsize=10, color=MID_GREY)
for r in range(6):
    ax.text(TX - 0.15, TY + r, f"nota {r}", ha="right", va="center", fontsize=10, color=DARK_GREY,
            fontweight="bold" if r == nota else "normal")
    casillas(TY + r, ej["tabla_wifi"][r], AZUL, x0=TX, resaltar=(r == nota))
ax.annotate("", xy=(X0 + 4 * ANCHO + 0.05, 1), xytext=(TX - 1.0, TY + nota),
            arrowprops={"arrowstyle": "->", "color": AZUL, "lw": 1.6, "connectionstyle": "arc3,rad=0.15"})
ax.text(TX + 4 * ANCHO + 0.15, TY + nota, f"<- este pasajero puso un {nota}:\n   la red recibe esta fila",
        fontsize=10, color=AZUL, va="center")
ax.text(TX - 0.9, TY + 6.6, "Cada dato con tabla funciona igual: se busca la fila de su valor\n"
                             "y esos números son los que entran en la red.", fontsize=10, color=MID_GREY, va="top")
save_shareable(fig, FIGS / "08_red_entrada.png", alt_text=(
    "Esquema con los 21 datos de un pasajero real y, junto a cada uno, las casillas con los números en que se convierte: "
    "13 valoraciones con 4 casillas cada una, 4 categorías con 4 casillas cada una, la distancia exacta con 16 casillas "
    "y 5 números sueltos. Total 89. A la derecha, la tabla del wifi con seis filas, una por nota, y la fila de la nota "
    "de este pasajero resaltada."))
plt.close(fig)
print("  08_red_entrada.png")

# ---------------------------------------------------------------------------------------------
# 5. La tabla del wifi antes y después de entrenar
# ---------------------------------------------------------------------------------------------
import torch

from playground_s6e10.cv import SEED

torch.manual_seed(SEED + 0)  # misma semilla que el fold 0: la tabla del wifi es lo primero que se crea
antes = torch.nn.Embedding(6, 4).weight.detach().numpy()
despues = ej["tabla_wifi"]
tasa = pd.read_csv(DATA / "train.csv", usecols=["Inflight wifi service", "satisfaction"])
tasa = tasa.groupby("Inflight wifi service")["satisfaction"].mean()

fig, ax = shareable_figure(
    title="El embedding empieza al azar y aquí casi no se mueve: aprende la capa que lo lee",
    subtitle="La tabla del wifi (una fila de 4 números por nota) antes y después de entrenar · red con embedding de la "
             "distancia, fold 0, 7 épocas",
    footer=PIE,
)
ax.set_xlim(0, 16.6)
ax.set_ylim(9.2, -2.2)
ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.05)
ANCHO = 0.95


def tabla(x0, valores, titulo, nota_abajo):
    ax.text(x0, -1.3, titulo, fontsize=14, color=AZUL, fontweight="bold")
    for r in range(6):
        ax.text(x0 - 0.15, r, f"nota {r}", ha="right", va="center", fontsize=12, color=DARK_GREY)
        for j, v in enumerate(valores[r]):
            ax.add_patch(Rectangle((x0 + j * ANCHO, r - 0.42), ANCHO - 0.06, 0.84, fc=SUAVE[AZUL], ec=AZUL, lw=1))
            ax.text(x0 + j * ANCHO + (ANCHO - 0.06) / 2, r, f"{v:+.2f}".replace(".", ","), ha="center", va="center",
                    fontsize=11.5, color=DARK_GREY)
    ax.text(x0 - 1.0, 6.2, nota_abajo, fontsize=11.5, color=MID_GREY, va="top", linespacing=1.4)


cambio = np.abs(despues - antes)
print(f"tabla del wifi: cambio medio {cambio.mean():.3f}, máximo {cambio.max():.3f}; "
      f"tamaño medio de los números al empezar {np.abs(antes).mean():.3f}")
tabla(1.6, antes, "1. Antes de entrenar: números al azar",
      "Nadie los ha elegido. Sirven de «código»:\ncada nota tiene uno distinto.")
tabla(8.6, despues, "2. Después de entrenar",
      "Cada número se ha corregido un poco en cada lote.\nCambio medio: " + coma(cambio.mean(), 2)
      + " (los números miden alrededor de 1).")
ax.annotate("", xy=(7.4, 2.5), xytext=(5.7, 2.5), arrowprops={"arrowstyle": "->", "color": MID_GREY, "lw": 2})
ax.text(6.55, 2.0, "unas 1.900\ncorrecciones", ha="center", va="bottom", fontsize=11, color=MID_GREY)
ax.text(13.4, -1.3, "% satisfechos (train)", fontsize=12, color=DARK_GREY, fontweight="bold")
for r in range(6):
    ax.text(14.4, r, f"{tasa[r] * 100:.0f} %", ha="center", va="center", fontsize=12, color=DARK_GREY)
ax.text(13.4, 6.2, "La red no recibe esta columna.\nSolo sabe si acierta o falla\ncon cada pasajero.", fontsize=11.5,
        color=MID_GREY, va="top", linespacing=1.4)
save_shareable(fig, FIGS / "08_red_embedding_antes_despues.png", alt_text=(
    "Dos tablas de 6 filas por 4 columnas con los números del embedding del wifi: a la izquierda los valores al azar "
    "con los que empieza la red y a la derecha los valores tras entrenar, que son casi los mismos. Al lado, el "
    "porcentaje de pasajeros satisfechos por nota en train, que la red no recibe."))
plt.close(fig)
print("  08_red_embedding_antes_despues.png")
