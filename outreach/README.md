# S6E10 · Paquete para el circuito de visibilidad

Preparado el 2026-10-23, después de publicar la **versión 3** del notebook en Kaggle. Es para Nicolás y para Claudio
(vault). Las explicaciones van en español y lo publicable en inglés.

**Todas las cifras de este paquete están medidas.** Las de CV salen de la ejecución del notebook, y coinciden con la
que hizo Kaggle el 23/10. Las del LB público son las de `kaggle competitions submissions`, comprobadas el 23/10. No se
pide el voto en ningún texto.

## Índice

| Pieza | Dónde | Estado |
|---|---|---|
| Hilo en X (`@nickpage100`), 8 posts | [abajo](#1-hilo-en-x) | listo para pegar |
| 3 gráficas 1600×900 con texto alternativo | [`charts/`](charts/) | listas |
| Post de LinkedIn | [abajo](#2-post-de-linkedin) | borrador para que Claudio lo pula |
| Case study del portfolio (EN/ES) | [abajo](#3-case-study-del-portfolio) | contenido; falta el `.ts` en `ferreret/portfolio` |
| Entrada de bitácora | [abajo](#4-entrada-de-bitácora) | borrador para el vault |
| Respuestas por si preguntan | [abajo](#5-por-si-preguntan-en-el-hilo) | — |

Las gráficas se regeneran con `uv run python outreach/make_charts.py`, desde la raíz del repo. Las dos primeras usan
el código del notebook publicado. La tercera es nueva y resume las tablas de las Parts 7.4 y 9.4.

## 1. Hilo en X

Formato del estreno del 28/09: introducción con la mejor gráfica, posts numerados con un hallazgo y sus cifras cada
uno, y el enlace solo en el último.

**Créditos sin `@`.** No he verificado la cuenta de X de nadie: Ιωάννης Σταμελιάς, Marc Maldonado Lorca, Sachith7 y
Rugved Bane. Si Claudio confirma alguna, se cambia el nombre por el `@`. Si no, se quedan con el nombre, igual que se
hizo con Marc en S6E9.

Todos los posts tienen 280 caracteres o menos, contando el enlace como 23.

**Introducción** · imagen [`charts/02_exact_distance.png`](charts/02_exact_distance.png), con el texto alternativo
de [`charts/02_exact_distance.txt`](charts/02_exact_distance.txt)

```text
Kaggle Playground S6E10: will an airline passenger say they are satisfied?

My new notebook solves it step by step and explains every concept the first time it appears.

The strongest hidden signal: each exact flight distance has its own satisfaction rate. 🧵
```

**1/**

```text
1/ Validation first. A LightGBM baseline scores 0.95882 in 5-fold CV and 0.95848 on the public leaderboard: a gap smaller than one standard deviation between folds.

Train and test are indistinguishable (adversarial AUC 0.4987), so the CV is the number to follow.
```

**2/** · imagen [`charts/01_zero_rating_heatmap.png`](charts/01_zero_rating_heatmap.png), con el texto alternativo
de [`charts/01_zero_rating_heatmap.txt`](charts/01_zero_rating_heatmap.txt)

```text
2/ A 0 rating is not "worse than 1". Passengers who gave the wifi a 0 are satisfied 89% of the time; with a 1, 2 or 3, under 40%.

One-hot encoding the ratings lifts a logistic regression from 0.92156 to 0.94859 AUC.

Ιωάννης Σταμελιάς spotted the 0 first in the forum.
```

**3/**

```text
3/ The generator copied the flight distances from the original survey: only 4 of 699,635 train rows have a distance the original never uses.

A leak-free target encoding of the exact distance: +0.00127 AUC in CV, +0.00125 on the leaderboard. Idea from Marc Maldonado Lorca (S6E9).
```

**4/**

```text
4/ LightGBM could partly see it alone, but its default max_bin = 255 merges the 3,474 distances into 255 bins.

With max_bin = 4095 the raw features gain +0.00084: two thirds of the encoding's gain, with no new column. A default can hide a pattern.
```

**5/**

```text
5/ There is a noise ceiling. A model trained on the original survey scores 0.995 AUC there and 0.954 on the synthetic passengers.

Almost half of the baseline's errors are confident predictions with the opposite label. No model will predict those.
```

**6/** · imagen [`charts/03_cv_vs_leaderboard.png`](charts/03_cv_vs_leaderboard.png), con el texto alternativo
de [`charts/03_cv_vs_leaderboard.txt`](charts/03_cv_vs_leaderboard.txt)

```text
6/ Know your noise floor. Changing only the seed moves the CV by −0.00009 and the leaderboard by +0.00012.

A teacher feature plus tuned parameters gained +0.00052 in CV, on every fold, and +0.00000 on the leaderboard. The final model keeps only what both sides confirm.
```

**7/** · último post, con el enlace

```text
7/ The full notebook, with what I tried that did not help and a glossary:
https://www.kaggle.com/code/nicolsbarcel/s6e10-step-by-step-every-concept-explained

Teacher-feature idea from Sachith7 and Rugved Bane. Written with AI assistance (Claude); analysis and decisions are mine.
```

**Antes de publicar**
- Hay que decidir si se fija en el perfil. Según la bitácora del 21/10, está fijado el hilo del asistente veterinario (20/10). Según el kit,
  se fija si es la mejor pieza hasta la fecha.
- La mención a la IA del 7/ es opcional en X. En el notebook y en el README ya se dice.

## 2. Post de LinkedIn

Es un borrador para que Claudio lo pula. Lo pega Nicolás. Lleva una imagen:
[`charts/03_cv_vs_leaderboard.png`](charts/03_cv_vs_leaderboard.png).

```text
Four changes improved my cross-validation score. The public leaderboard confirmed only one.

This month's Kaggle Playground (S6E10) asks whether an airline passenger will say they are satisfied. I published a notebook that solves it step by step and explains every concept the first time it appears.

Three things I would tell anyone starting a tabular project:

1. Validate before you model. My baseline scored 0.95882 in 5-fold CV and 0.95848 on the public leaderboard. With a validation that tracks the leaderboard, you can decide what to keep without spending submissions.

2. Look at the data before you tune. The synthetic data copied the flight distances of the original survey, and each exact distance carries its own satisfaction rate. A leak-free target encoding of that one value added +0.00127 AUC in CV and +0.00125 on the leaderboard: more than any tuning I tried.

3. Know your noise floor. Retraining the same model with another random seed moved the leaderboard by 0.00012. A teacher feature and tuned parameters gained +0.00052 in CV, on every fold, and nothing on the leaderboard. So the final model keeps only the change both sides confirm.

The notebook also covers what did not help, and why.

Notebook: https://www.kaggle.com/code/nicolsbarcel/s6e10-step-by-step-every-concept-explained
Code: https://github.com/ferreret/kaggle-playground-s6e10

Built with AI assistance (Claude); the analysis direction and the decisions are mine.

#MachineLearning #Kaggle #DataScience #LightGBM
```

## 3. Case study del portfolio

**Algo que decidir antes.** En `ferreret/portfolio` no hay ninguna entrada de Kaggle; S6E9 tampoco está. Hay dos
opciones:

- **(a)** una entrada para S6E10, como el resto de proyectos de `data/projects/`;
- **(b)** una entrada «Kaggle Playground notebooks» que reúna S6E9 y S6E10 y vaya creciendo.

Yo me inclino por **(b)**, porque el objetivo es la serie de notebooks hacia Expert y no un episodio suelto. El
contenido de abajo sirve para cualquiera de las dos.

Los campos siguen la forma de `ProjectItem`. El `.ts` y la imagen de portada (que puede ser
`charts/02_exact_distance.png` en `.webp`) quedan para la sesión del portfolio.

**EN**

- **title**: Kaggle Playground S6E10 — Predicting Airline Satisfaction
- **description**: A public Kaggle notebook that predicts whether an airline passenger will say they are satisfied,
  written step by step for readers who want to understand a complete tabular ML workflow — every concept explained the
  first time it appears.
- **role**: Data scientist & author
- **timeline**: October 2026 · v1 published on day 2, v3 on day 23
- **links**: notebook `https://www.kaggle.com/code/nicolsbarcel/s6e10-step-by-step-every-concept-explained` · repo
  `https://github.com/ferreret/kaggle-playground-s6e10`
- **problem**: Kaggle's October 2026 Playground competition gives 699,635 synthetic passenger records, each with 21
  features (who the passenger is, the trip, 13 service ratings and two delays), and asks for the probability that the
  passenger declared themselves satisfied, scored with ROC AUC. The data was generated by a deep-learning model from a
  real 129,880-passenger survey. The goal was not the leaderboard position but a notebook that other people can read
  from top to bottom and reuse.
- **solution**: Validation came before modelling: a stratified 5-fold CV that tracks the public leaderboard (0.95882
  against 0.95848), with adversarial validation showing train and test are indistinguishable. The EDA found that a `0`
  rating means "satisfied" for digital services, which is worth +0.027 AUC to a logistic regression once the ratings
  are one-hot encoded. Comparing the synthetic data with the original showed that the generator copied the exact flight
  distances, each with its own satisfaction rate; a leak-free target encoding of that value lifted LightGBM to 0.96009 ±
  0.00055 in CV and 0.95973 on the public leaderboard. SHAP explains the model in the language of the problem. A final
  section measures the noise floor and reports, with the same honesty, the ideas that improved the CV but not the
  leaderboard.
- **businessMetrics**:
  - Training rows: 699,635
  - Final CV AUC: 0.96009 ± 0.00055
  - Public leaderboard AUC: 0.95973
  - Baseline CV–leaderboard gap: 0.00034 (under 1 std)
  - Ideas tested against the final model: 6, plus a seed control
- **techStack**: ML: LightGBM, scikit-learn, SHAP values via LightGBM `pred_contrib` · Data: pandas, NumPy · Charts:
  Matplotlib · Tooling: uv, Kaggle API, Jupyter
- **lessonsLearned**:
  - The biggest gain came from the data, not from tuning: one exact value copied by the generator was worth more than
    the hyperparameter search.
  - A default can hide a pattern: LightGBM's `max_bin = 255` merged 3,474 distances into 255 bins; raising it recovered
    two thirds of the gain with no new feature.
  - Measure the noise floor before believing a gain: a new random seed alone moved the leaderboard by 0.00012.
  - Gains of 0.0003 cannot be confirmed by the CV or the public leaderboard alone; the final model keeps only what both
    agree on.
  - Publish early and improve the same notebook: version 1 was public on day 2, and later versions add findings rather
    than new notebooks.

**ES**

- **title**: Kaggle Playground S6E10 — Predicción de la satisfacción de pasajeros
- **description**: Un notebook público de Kaggle que predice si un pasajero de avión dirá que está satisfecho, escrito
  paso a paso para quien quiera entender un flujo completo de ML tabular: cada concepto se explica la primera vez que
  aparece.
- **role**: Científico de datos y autor
- **timeline**: Octubre de 2026 · v1 publicada el día 2, v3 el día 23
- **problem**: La competición Playground de octubre de 2026 de Kaggle da 699.635 registros sintéticos de pasajeros,
  con 21 variables (quién es el pasajero, el viaje, 13 valoraciones de servicios y dos retrasos), y pide la
  probabilidad de que el pasajero se declare satisfecho, evaluada con ROC AUC. Los datos los generó un modelo de
  aprendizaje profundo a partir de una encuesta real de 129.880 pasajeros. El objetivo no era el puesto en la
  clasificación, sino un notebook que otras personas puedan leer de arriba abajo y reutilizar.
- **solution**: La validación fue antes que el modelo: una validación cruzada estratificada de 5 folds que sigue al
  leaderboard público (0,95882 frente a 0,95848), con una validación adversarial que muestra que train y test son
  indistinguibles. El EDA encontró que un `0` en las valoraciones significa «satisfecho» en los servicios digitales, lo
  que vale +0,027 de AUC a una regresión logística si las valoraciones se codifican one-hot. Comparar los datos
  sintéticos con el original mostró que el generador copió las distancias de vuelo exactas, cada una con su propia
  tasa de satisfacción; un target encoding sin fuga de ese valor llevó a LightGBM a 0,96009 ± 0,00055 en CV y 0,95973
  en el leaderboard público. SHAP explica el modelo en el lenguaje del problema. Una sección final mide el suelo de
  ruido y cuenta, con la misma honestidad, las ideas que mejoraron la CV pero no el leaderboard.
- **businessMetrics**:
  - Filas de entrenamiento: 699.635
  - AUC final en CV: 0,96009 ± 0,00055
  - AUC en el leaderboard público: 0,95973
  - Diferencia CV–leaderboard del baseline: 0,00034 (menos de 1 desviación típica)
  - Ideas probadas contra el modelo final: 6, más un control de semilla
- **techStack**: ML: LightGBM, scikit-learn, valores SHAP con `pred_contrib` de LightGBM · Datos: pandas, NumPy ·
  Gráficas: Matplotlib · Herramientas: uv, API de Kaggle, Jupyter
- **lessonsLearned**:
  - La mayor mejora vino de los datos, no del ajuste: un solo valor exacto copiado por el generador valió más que
    la búsqueda de hiperparámetros.
  - Un valor por defecto puede esconder un patrón: el `max_bin = 255` de LightGBM juntó 3.474 distancias en 255
    tramos; subirlo recuperó dos tercios de la mejora sin ninguna variable nueva.
  - Medir el suelo de ruido antes de creerse una mejora: cambiar solo la semilla movió el leaderboard 0,00012.
  - Una mejora de 0,0003 no la confirma ni la CV ni el leaderboard público por separado; el modelo final se queda solo
    con lo que confirman los dos.
  - Publicar pronto y mejorar el mismo notebook: la versión 1 fue pública el día 2, y las siguientes añaden hallazgos
    en lugar de abrir notebooks nuevos.

## 4. Entrada de bitácora

Borrador para `01_Projects/Productos/Kaggle_HuggingFace/Bitacora.md`. Claudio lo adapta a su estilo.

```markdown
- **2026-10-23**: 🚀 **S6E10: versión 3 del notebook publicada y envíos finales elegidos.** Sesión de tarde en el PC Linux con Claude Code. Cierra el pendiente del 08/10 (el borrador de la v3 seguía sin publicar).
  - 📓 **Notebook**: la v3 sustituye al principal (commit [`f77ede2`](https://github.com/ferreret/kaggle-playground-s6e10/commit/f77ede2), con push) y se sube con `kaggle kernels push` como **versión 3 del mismo notebook**. Kaggle la ejecutó de cero sin errores, y todas las cifras de CV coinciden con la ejecución local. Novedades: `max_bin` 4095 (+0,00084 sin columnas nuevas, dos tercios de la mejora del TE), un control con la semilla 43 como suelo de ruido, tres ideas que no lo superan (filas del original, con marca y sin ella, y los ceros como NaN) y dos que mejoran la CV pero no el LB (teacher y parámetros ajustados: +0,00052 en CV, +0,00000 en LB).
  - 📄 **README**: tabla completa de CV y LB y créditos nuevos (karttikjangid05, Sachith7, Rugved Bane).
  - 🏁 **Envíos finales**, marcados en la web: **04** (mezcla 0,8 LightGBM + 0,2 red; CV 0,96074, LB 0,95995) y **02** (LightGBM + TE de la distancia; CV 0,96009, LB 0,95973). Cierre: 31/10 23:59 UTC. Puesto el 23/10: 1337 de 2802. El notebook tenía 2 votos.
  - 🔑 La CLI de Kaggle se volvió a autenticar con `kaggle auth login` (OAuth). Hay que lanzarla desde el repo, con `uv run --directory …`.
  - ⏭️ **Queda**: el circuito de visibilidad. El paquete está en `outreach/` del repo: hilo de X, gráficas con texto alternativo, LinkedIn, case study y esta entrada.
```

## 5. Por si preguntan en el hilo

- **¿Por qué el modelo final no es el mejor del LB?** El mejor LB público es la mezcla con una red neuronal (envío 04:
  CV 0,96074, LB 0,95995). Va como uno de los dos envíos finales, pero no está en el notebook. Su ventaja en el LB
  (+0,00022 sobre el 02) apenas supera lo que mueve una semilla (+0,00012). El notebook se queda con lo que confirman
  la CV y el LB a la vez.
- **¿Qué es la «teacher feature»?** La predicción de un LightGBM entrenado solo con la encuesta original, usada como
  una columna más. No puede haber fuga porque ese modelo nunca vio una etiqueta sintética.
- **¿Probaste añadir las filas del original?** Sí: −0,00015 en CV, peor en los 5 folds. Con una columna que marca
  las filas reales, +0,00008 en 3 de 5 folds, que no se distingue del ruido.
- **¿Por qué el 0 de wifi empuja más que un 5?** SHAP describe el modelo, no el mundo. El 0 probablemente marca un
  tipo de vuelo o de pasajero que no está en las columnas. Lo honesto es decir que esos pasajeros están satisfechos
  mucho más a menudo, por razones que los datos no muestran.
- **¿Cuánto tarda?** 44 minutos en un equipo de 16 núcleos. En Kaggle, según el log de la versión 3, solo las
  celdas de validación cruzada suman unos 55 minutos.
