# Kaggle Playground S6E10 — Predicting Airline Satisfaction

Work for the [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10) competition
(October 2026): predict whether an airline passenger declares themselves satisfied. Metric: ROC AUC.

The deliverable is one public notebook that solves the problem from scratch and explains every concept the first
time it appears: [`notebooks/s6e10-step-by-step.ipynb`](notebooks/s6e10-step-by-step.ipynb).

**Main notebook on Kaggle:** [S6E10 Step by Step: Every Concept Explained](https://www.kaggle.com/code/nicolsbarcel/s6e10-step-by-step-every-concept-explained)

## Results

5-fold stratified CV (seed 42), same folds for every model.

| Model | CV AUC (mean ± std) | Public LB |
|---|---|---|
| Logistic regression, ratings as numbers | 0.92156 ± 0.00044 | — |
| Logistic regression, ratings one-hot encoded | 0.94859 ± 0.00050 | — |
| LightGBM, raw features | 0.95882 ± 0.00063 | 0.95848 |
| LightGBM, raw features, `max_bin` 4095 | 0.95966 ± 0.00062 | — |
| **LightGBM + target encoding of the exact flight distance** (final model) | **0.96009 ± 0.00055** | **0.95973** |

Ideas tested against the final model (Part 9 of the notebook). Gains are paired, fold by fold, against the final
model's CV of 0.96009 and public LB of 0.95973.

| Change | CV AUC (mean ± std) | CV gain (folds that improve) | Public LB | LB gain |
|---|---|---|---|---|
| Control: same recipe, seed 43 | 0.96000 ± 0.00061 | −0.00009 (1 of 5) | 0.95985 | +0.00012 |
| + original rows in training | 0.95993 ± 0.00056 | −0.00015 (0 of 5) | — | — |
| + original rows, with an `is_original` flag | 0.96016 ± 0.00057 | +0.00008 (3 of 5) | — | — |
| `0` ratings as missing | 0.96007 ± 0.00056 | −0.00001 (2 of 5) | — | — |
| + teacher feature (model trained on the original) | 0.96036 ± 0.00060 | +0.00027 (5 of 5) | 0.95960 | −0.00013 |
| Tuned LightGBM parameters | 0.96033 ± 0.00061 | +0.00024 (5 of 5) | 0.95957 | −0.00016 |
| + teacher feature, tuned parameters | 0.96061 ± 0.00056 | +0.00052 (5 of 5) | 0.95973 | +0.00000 |

## Main findings

- **CV tracks the leaderboard**: 0.95882 in CV against 0.95848 on the public leaderboard, a gap smaller than one
  standard deviation between folds. The next model gained +0.00127 in CV and +0.00125 on the leaderboard. Train and
  test are indistinguishable (adversarial validation AUC 0.4987).
- **A rating of `0` is not a bad score**: with `Inflight wifi service = 0`, 89% of passengers are satisfied.
  The ratings are not monotonic, which costs a linear model 0.027 AUC unless they are one-hot encoded.
- **The generator copied the flight distances from the original dataset**: only 4 of 699,635 train rows have a
  distance the original does not use. Each exact distance has its own satisfaction rate, and a leak-free target
  encoding of it adds +0.00127 AUC in CV, in all five folds. Raising LightGBM's `max_bin` from 255 to 4095, so that
  every distance gets its own bin, recovers two thirds of that gain with the raw features alone.
- **There is a noise ceiling**: a model trained only on the original data scores 0.995 on the original and 0.954 on
  the synthetic train. Almost half of the baseline's errors are confident predictions with the opposite label.
- **The noise floor is about 0.0001 on both sides**: changing only the seed moves the CV by −0.00009 and the public
  leaderboard by +0.00012. Original rows, a flag for them and zeros as missing do not beat it.
- **Small CV gains the leaderboard does not confirm**: a teacher feature and tuned parameters gain +0.00052 together
  in CV, on every fold, and +0.00000 on the public leaderboard. The final model stays the one whose gain both sides
  confirm; the private leaderboard will say which side was right.

## Repository layout

```
notebooks/s6e10-step-by-step.ipynb   the notebook that gets published
notebooks/kernel-metadata.json       Kaggle publishing settings (title, inputs)
experiments/                          numbered scratch experiments (in Spanish), not published
src/playground_s6e10/plots.py         shareable chart style
data/                                 Kaggle data (git-ignored)
```

## Reproduce

```bash
uv sync
uv run kaggle competitions download -c playground-series-s6e10 -p data && unzip -o data/playground-series-s6e10.zip -d data
uv run kaggle datasets download arseniyshutko/binary-aviation-satisfaction-129k -p data/original --unzip
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace s6e10-step-by-step.ipynb
```

Python 3.12; exact package versions are pinned in `uv.lock`.

## Data

- Competition data: [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10/data).
- Original dataset the synthetic data was generated from:
  [Binary Aviation Satisfaction (129k)](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k) (CC0).

## Credits

- The idea of looking for exact values copied from the original dataset comes from Marc Maldonado Lorca's
  [S6E9 discussion](https://www.kaggle.com/competitions/playground-series-s6e9/discussion/742385).
- The `0` rating as "not applicable" was reported first in the S6E10 forum by Ιωάννης Σταμελιάς
  ([discussion](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/744809)); the numbers in the
  notebook were computed independently and agree with his.
- Adding the original rows was tested after karttikjangid05's
  [discussion](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745098), with the same result.
- The teacher feature was shared by Sachith7
  ([discussion](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745908)) and Rugved Bane
  ([discussion](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745555)).

## Acknowledgements

This work was done with the assistance of Claude (Anthropic), an AI model. Analysis direction, review and final
decisions are the author's.
