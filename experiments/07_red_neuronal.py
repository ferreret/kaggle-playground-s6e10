"""07 · Una red neuronal pequeña, sola y mezclada con LightGBM.

Mismos folds que el notebook. Dos redes iguales salvo por una cosa:

  A. Sin la distancia exacta: la distancia entra solo como número.
  B. Con la distancia exacta: además, cada distancia tiene su propio «embedding» (una fila de 16
     números que la red aprende). Es la versión «red neuronal» del hallazgo de la parte 7.

Después se mezcla cada red con el mejor LightGBM (target encoding de la distancia, CV 0,96009).

Cómo entra cada columna en la red:
  - 13 valoraciones (0..5) y 4 categóricas: un embedding por columna. Así la nota 0 y la nota 5 no
    están obligadas a ser «poco» y «mucho»: cada nota tiene su propia representación.
  - Edad, distancia y los dos retrasos: como números, reescalados (media 0, desviación 1).
  - Retraso de llegada ausente: se rellena con 0 y se añade una columna 0/1 que lo marca.

Uso: uv run --group nn python experiments/07_red_neuronal.py
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from playground_s6e10.cv import DIST, OUT, SEED, load, make_folds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH, MAX_EPOCHS, PACIENCIA, LR = 2048, 40, 4, 2e-3
DEP, ARR = "Departure Delay in Minutes", "Arrival Delay in Minutes"

train, test, orig, y, y_orig, FEATURES, CATEGORICAL, RATINGS = load()
folds = make_folds(train, y)
REF = json.loads((OUT / "04_distancia_te.json").read_text())["te"]["fold_scores"]
print(f"dispositivo: {DEVICE} | torch {torch.__version__}", flush=True)

# Columnas que entran como categoría (índices enteros) y sus tamaños
CAT_COLS = RATINGS + CATEGORICAL
CARDS = [6] * len(RATINGS) + [len(train[c].cat.categories) for c in CATEGORICAL]


def categorias(df):
    return np.column_stack([df[c].to_numpy() for c in RATINGS] + [df[c].cat.codes.to_numpy() for c in CATEGORICAL])


def numericas(df):
    llegada = df[ARR]
    return np.column_stack([
        df["Age"], np.log1p(df[DIST]), np.log1p(df[DEP]), np.log1p(llegada.fillna(0)), llegada.isna().astype(float),
    ]).astype(np.float32)


cat_train, cat_test = categorias(train), categorias(test)
num_train, num_test = numericas(train), numericas(test)


class Red(nn.Module):
    def __init__(self, n_dist):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, 4) for c in CARDS])
        self.dist = nn.Embedding(n_dist, 16) if n_dist else None
        entrada = 4 * len(CARDS) + num_train.shape[1] + (16 if n_dist else 0)
        self.capas = nn.Sequential(
            nn.Linear(entrada, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, cat, num, dist):
        partes = [emb(cat[:, i]) for i, emb in enumerate(self.embs)] + [num]
        if self.dist is not None:
            partes.append(self.dist(dist))
        return self.capas(torch.cat(partes, dim=1)).squeeze(1)


def predecir(red, cat, num, dist):
    red.eval()
    salidas = []
    with torch.no_grad():
        for i in range(0, len(cat), 16384):
            salidas.append(torch.sigmoid(red(cat[i:i + 16384], num[i:i + 16384], dist[i:i + 16384])))
    return torch.cat(salidas).cpu().numpy()


def entrenar_fold(k, tr_idx, va_idx, con_distancia):
    torch.manual_seed(SEED + k)
    # Reescalado con la media y la desviación de las filas de entrenamiento del fold
    media, desv = num_train[tr_idx].mean(axis=0), num_train[tr_idx].std(axis=0) + 1e-6
    media[-1], desv[-1] = 0.0, 1.0  # la columna 0/1 de «ausente» se deja como está
    a_gpu = lambda a, t: torch.as_tensor(a, dtype=t, device=DEVICE)
    num = {n: a_gpu((x - media) / desv, torch.float32) for n, x in
           [("tr", num_train[tr_idx]), ("va", num_train[va_idx]), ("te", num_test)]}
    cat = {n: a_gpu(x, torch.long) for n, x in [("tr", cat_train[tr_idx]), ("va", cat_train[va_idx]), ("te", cat_test)]}
    # Índice de cada distancia: las que tienen menos de 5 filas en el entrenamiento comparten el índice 0
    cuenta = train[DIST].iloc[tr_idx].value_counts()
    indice = {d: i + 1 for i, d in enumerate(cuenta[cuenta >= 5].index)}
    dist = {n: a_gpu(s.map(indice).fillna(0).to_numpy(), torch.long) for n, s in
            [("tr", train[DIST].iloc[tr_idx]), ("va", train[DIST].iloc[va_idx]), ("te", test[DIST])]}
    y_tr = a_gpu(y[tr_idx], torch.float32)

    red = Red(len(indice) + 1 if con_distancia else 0).to(DEVICE)
    opt = torch.optim.AdamW(red.parameters(), lr=LR, weight_decay=1e-5)
    plan = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    perdida = nn.BCEWithLogitsLoss()
    mejor, mejor_estado, sin_mejorar, historia = 0.0, None, 0, []
    for epoca in range(MAX_EPOCHS):
        red.train()
        orden, total = torch.randperm(len(tr_idx), device=DEVICE), 0.0
        for i in range(0, len(orden), BATCH):
            lote = orden[i:i + BATCH]
            opt.zero_grad()
            error = perdida(red(cat["tr"][lote], num["tr"][lote], dist["tr"][lote]), y_tr[lote])
            error.backward()
            opt.step()
            total += error.item() * len(lote)
        auc = roc_auc_score(y[va_idx], predecir(red, cat["va"], num["va"], dist["va"]))
        historia.append({"epoca": epoca + 1, "perdida_entrenamiento": total / len(tr_idx), "auc_validacion": auc})
        plan.step(auc)
        if auc > mejor:
            mejor, sin_mejorar = auc, 0
            mejor_estado = {n: v.detach().clone() for n, v in red.state_dict().items()}
        else:
            sin_mejorar += 1
            if sin_mejorar >= PACIENCIA:
                break
    red.load_state_dict(mejor_estado)
    if con_distancia and k == 0:  # ejemplo para las gráficas: los 89 números del primer pasajero de validación
        red.eval()
        with torch.no_grad():
            np.savez(OUT / "07_ejemplo_entrada.npz", fila=int(va_idx[0]), columnas=np.array(CAT_COLS),
                     embeddings=torch.stack([e(cat["va"][0, j]) for j, e in enumerate(red.embs)]).cpu().numpy(),
                     numericas=num["va"][0].cpu().numpy(), distancia=red.dist(dist["va"][0]).cpu().numpy(),
                     tabla_wifi=red.embs[0].weight.cpu().numpy())
    return (predecir(red, cat["va"], num["va"], dist["va"]), predecir(red, cat["te"], num["te"], dist["te"]),
            historia, sum(p.numel() for p in red.parameters()))


resultados = {}
for clave, nombre, con_distancia in [("sin_distancia", "A. Red sin la distancia exacta", False),
                                     ("con_distancia", "B. Red con embedding de la distancia exacta", True)]:
    print(f"\n{nombre}:", flush=True)
    oof, test_pred, scores, historias, inicio = np.zeros(len(train)), np.zeros(len(test)), [], [], time.time()
    for k, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.time()
        oof[va_idx], pred_te, historia, n_param = entrenar_fold(k, tr_idx, va_idx, con_distancia)
        test_pred += pred_te / len(folds)
        scores.append(roc_auc_score(y[va_idx], oof[va_idx]))
        historias.append(historia)
        mejor_epoca = int(np.argmax([h["auc_validacion"] for h in historia])) + 1
        print(f"  fold {k}: {scores[-1]:.5f} (LightGBM {REF[k]:.5f}, dif {scores[-1] - REF[k]:+.5f}) | "
              f"mejor época {mejor_epoca} de {len(historia)} | {time.time() - t0:.0f}s", flush=True)
    print(f"{nombre}: CV {np.mean(scores):.5f} ± {np.std(scores):.5f} | parámetros {n_param:,} | {time.time() - inicio:.0f}s")

    # Mezcla con LightGBM: w * red + (1 - w) * LightGBM, sobre las predicciones fuera de fold
    lgbm_oof = np.load(OUT / "04_oof_te.npy")
    mezcla = {}
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        por_fold = [roc_auc_score(y[va], w * oof[va] + (1 - w) * lgbm_oof[va]) for _, va in folds]
        mezcla[float(w)] = {"cv": float(np.mean(por_fold)), "dif": float(np.mean(por_fold) - np.mean(REF)),
                            "folds_mejoran": int((np.array(por_fold) > np.array(REF)).sum()),
                            "dif_min": float((np.array(por_fold) - np.array(REF)).min())}
    print("  mezcla (peso de la red -> CV, dif sobre LightGBM solo, folds que mejoran):")
    for w, m in mezcla.items():
        print(f"    {w:.1f} -> {m['cv']:.5f}  {m['dif']:+.5f}  {m['folds_mejoran']}/5  (peor fold {m['dif_min']:+.5f})")
    print(f"  correlación entre las predicciones de la red y las de LightGBM: {np.corrcoef(oof, lgbm_oof)[0, 1]:.4f}")

    resultados[clave] = {"fold_scores": scores, "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores)),
                         "n_parametros": n_param, "historias": historias, "mezcla": mezcla}
    np.save(OUT / f"07_oof_{clave}.npy", oof)
    np.save(OUT / f"07_test_{clave}.npy", test_pred)
    (OUT / "07_red_neuronal.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
