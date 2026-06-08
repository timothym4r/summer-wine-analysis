# UTEA Wine Review Rating Prediction

This repository contains experiments for predicting Wine Enthusiast review scores from review text. The project currently supports two related tasks:

- **Regression:** predict the numeric `points` score.
- **Ordered classification:** convert `points` into ordered rating bins and predict the bin.

The code is organized so classical, interpretable text baselines can be compared against frozen sentence-embedding models and optional future LLM/transformer extensions.

## Data

The default dataset path is:

```text
data/WineEnthusiast-data/winemag-data-130k-v2.csv
```

The expected columns are:

- `description`: wine review text.
- `points`: numeric Wine Enthusiast rating.

Data loading is handled by `modules/data.py`. It drops rows with missing text or target values, coerces `points` to numeric, and removes duplicate descriptions by default.

The project `.gitignore` excludes `data/`, `outputs/`, and large generated artifacts, so local results and datasets are not meant to be committed.

## Project Layout

```text
run_experiments.py              Main classical baseline runner
run_embedding_experiments.py    Frozen sentence-embedding experiment runner
analyze_traditional_results.py  Summary and prediction-level error analysis

modules/
  binning.py                    Rating distribution summaries and ordered bins
  config.py                     Shared paths, random seed, model/vectorizer defaults
  data.py                       Data loading and train/valid/test splitting
  evaluation.py                 Regression/classification metrics
  experiments.py                Classical baseline experiment loops
  features.py                   TF-IDF, count/indicator, and text-stat pipelines
  models.py                     Classical sklearn model registry
  interpretability.py           Coefficient-based feature importance reports
  error_analysis.py             Prediction files and confusion matrices
  embedding_experiments.py      Frozen embedding experiments and outputs
  transformer_models.py         Sentence embedding cache and future fine-tuning stubs
  llm_features.py               Optional LLM attribute extraction stub
  llm_classifier.py             Optional few-shot LLM prediction helpers
```

## Preprocessing And Splits

Text preprocessing is intentionally light:

- whitespace is normalized;
- punctuation is preserved;
- case is preserved by default.

The split logic is centralized in `train_valid_test_split`:

- test size: `0.2`
- validation size: `0.1`
- random seed: `42`

For model fitting, train and validation are usually recombined into `train_eval`; final metrics are reported on the held-out test set.

## Rating Bins

Classification uses ordered bins created by `make_rating_bins` in `modules/binning.py`.

The current default is quantile binning with four bins:

```text
-inf-86
86-88
88-91
91-inf
```

Classification metrics treat these as ordered classes when computing:

- mean absolute class error;
- quadratic weighted kappa.

## Classical Feature Families

Classical features live in `modules/features.py`.

### TF-IDF

Configured in `TFIDF_CONFIG`:

```python
max_features = 50000
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
sublinear_tf = True
```

This produces unigram and bigram TF-IDF features.

### Count

Configured in `COUNT_VECTORIZER_CONFIG`:

```python
max_features = 50000
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
```

This produces unigram and bigram raw count features.

### Indicator N-Grams

Added as binary presence/absence versions of the count features. These use the same default n-gram range as count features:

```python
ngram_range = (1, 2)
binary = True
```

Each feature is `1` if the word or n-gram appears in the review and `0` otherwise. These results are labeled as:

```text
feature_family = indicator
```

Preposition/conjunction removal is controlled by a run argument. When enabled, it applies consistently to all n-gram methods: TF-IDF, count, and indicator. Negation-bearing terms such as `not`, `no`, `nor`, `never`, and `without` are preserved.

### Text Stats

Simple numeric review features:

- character count;
- word count;
- average word length;
- sentence count;
- comma/period/exclamation/question/semicolon counts;
- total punctuation count.

These are standardized before modeling.

## Classical Models

Model definitions live in `modules/models.py`; hyperparameters live in `modules/config.py`.

### Regression Models

- `dummy_mean`
- `ridge`
- `linear_svr`
- `decision_tree`
- `random_forest`
- `hist_gradient_boosting`

Current experiment routing in `modules/experiments.py`:

- TF-IDF: `dummy_mean`, `ridge`, `linear_svr`
- Count: `ridge`, `linear_svr`
- Indicator n-grams: `ridge`, `linear_svr`, `decision_tree`
- Text stats: all non-dummy regression models

### Classification Models

- `dummy_most_frequent`
- `logistic_regression`
- `linear_svc`
- `decision_tree`
- `random_forest`
- `hist_gradient_boosting`

Current experiment routing:

- TF-IDF: `dummy_most_frequent`, `logistic_regression`, `linear_svc`
- Count: `logistic_regression`, `linear_svc`
- Indicator n-grams: `logistic_regression`, `linear_svc`, `decision_tree`
- Text stats: all non-dummy classification models

The decision tree config is currently:

```python
max_depth = 30
min_samples_leaf = 5
```

## Metrics

Regression uses:

- MAE
- RMSE
- R2

Classification uses:

- accuracy
- macro F1
- weighted F1
- confusion matrix
- mean absolute class error
- quadratic weighted kappa

## Running Experiments

Install the core Python dependencies in your environment:

```bash
pip install pandas numpy scikit-learn tqdm
```

Run the classical baseline suite:

```bash
python3 run_experiments.py
```

Useful options:

```bash
python3 run_experiments.py --data path/to/data.csv --output-dir outputs
python3 run_experiments.py --skip-interpretability
python3 run_experiments.py --standardize-target
python3 run_experiments.py --remove-prepositions-conjunctions
```

`--standardize-target` applies only to regression. It trains regression models on z-scored `points`, then inverse-transforms predictions before computing MAE, RMSE, and R2 so reported metrics remain in the original points scale.

`--remove-prepositions-conjunctions` applies to all n-gram feature methods: TF-IDF, count, and indicator. If it is not set, none of those n-gram methods apply this removal.

Generate summaries and focused error-analysis files from classical outputs:

```bash
python3 analyze_traditional_results.py
```

This reads existing results, writes summary files under `outputs/summary/`, and optionally refits selected TF-IDF models to save row-level prediction errors under `outputs/error_analysis/`.

## Frozen Sentence Embeddings

Embedding experiments are in `run_embedding_experiments.py` and `modules/embedding_experiments.py`.

Default embedding models are configured in `EMBEDDING_MODELS`:

```python
minilm = sentence-transformers/all-MiniLM-L6-v2
mpnet = sentence-transformers/all-mpnet-base-v2
```

Install optional embedding dependencies:

```bash
pip install sentence-transformers
```

Run Stage 1 embedding experiments:

```bash
python3 run_embedding_experiments.py
```

Run a smaller smoke test:

```bash
python3 run_embedding_experiments.py --models minilm --sample-size 1000 --no-nn --output-dir outputs/advanced_embeddings_smoke
```

Embedding outputs are written under the selected output directory:

```text
results/
  embedding_regression_results.csv
  embedding_classification_results.csv
error_analysis/
  *_regression_predictions.csv
  *_classification_predictions.csv
  *_classification_confusion_matrix.csv
summary/
  embedding_stage1_summary.json
  embedding_stage1_summary.md
  embedding_interpretability_summary.csv
cache/
  *.npz
  *.json
```

## Interpretability

Classical interpretability reports are generated by `modules/interpretability.py` during `run_experiments.py` unless `--skip-interpretability` is passed.

Current report paths:

```text
outputs/interpretability/regression_feature_importance.csv
outputs/interpretability/classification_feature_importance.csv
```

These reports are coefficient-based, so they are available for linear models such as Ridge and logistic regression. They should be interpreted as learned associations, not causal explanations.

Tree models are not currently included in the interpretability report. If tree feature importance is needed later, add a separate extractor for `feature_importances_`.

## Current Known Results

The existing `outputs/regression_results.csv` and `outputs/classification_results.csv` may lag behind current code if new feature families or arguments have been added since the last full run.

Best existing classical regression result:

```text
tfidf + ridge
MAE  = 1.3000
RMSE = 1.6530
R2   = 0.7135
```

Best existing classical classification result by macro F1:

```text
tfidf + logistic_regression
accuracy = 0.5913
macro F1 = 0.5931
weighted F1 = 0.5881
quadratic weighted kappa = 0.7536
```

Historical targeted run for the earlier unigram indicator decision trees:

```text
Regression: indicator_unigram + decision_tree
MAE  = 2.0734
RMSE = 2.6347
R2   = 0.2721
features = 18128
```

```text
Classification: indicator_unigram + decision_tree
accuracy = 0.4374
macro F1 = 0.4255
weighted F1 = 0.4313
mean absolute class error = 0.7331
quadratic weighted kappa = 0.4902
features = 18114
```

The current code now uses the broader `indicator` family with 1-grams and 2-grams plus multiple linear/tree models. Preposition/conjunction filtering is opt-in through `--remove-prepositions-conjunctions`. To refresh the main result CSVs, rerun:

```bash
python3 run_experiments.py
```

## Optional LLM And Transformer Work

The repository has placeholders for future extensions:

- `modules/llm_features.py`: extracts structured wine attributes with an explicitly provided OpenAI client; defaults to mock zero attributes when no client is passed.
- `modules/llm_classifier.py`: few-shot classification/regression helper functions; requires an explicit LLM client.
- `modules/transformer_models.py`: sentence embedding support is active; Hugging Face fine-tuning functions are intentionally stubs.

No external LLM calls happen in the baseline scripts unless a client is explicitly wired in later.

## Notes For Future Codex Sessions

Start with these files:

- `modules/experiments.py` for what actually runs in the classical pipeline.
- `modules/config.py` for defaults and hyperparameters.
- `modules/features.py` for feature definitions.
- `modules/models.py` for model registration.
- `modules/evaluation.py` for metric definitions.

Important implementation details:

- Current TF-IDF and count features use 1-grams and 2-grams.
- Indicator features also use 1-grams and 2-grams, but are binary presence/absence features.
- Preposition/conjunction filtering is opt-in and applies to all n-gram feature families together.
- Negation words such as `not` should not be casually removed if stopword filtering is changed later.
- `--standardize-target` changes regression training targets only; evaluation is still reported on the original point scale.
- Existing output CSVs may lag behind code changes; check timestamps or rerun before relying on them.
- `data/` and `outputs/` are ignored by git, so missing local artifacts are expected on a fresh clone.
