# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This is a graded M2 (Master 2, Génie Informatique) Machine Learning exam project ("Sujet A — Détection de fraude par carte de crédit"), not a production system. The full assignment brief is `Projet_Examen_Machine_Learning_M2.pdf` (untracked, not for the repo). Key constraints from that brief that should shape any work here:

- **Deliverables**: an end-to-end-executable Jupyter notebook, an 8–12 page written report (PDF, separate from this repo), the Git repo itself, and a 15-minute oral defense. The notebook, report, and repo must tell the same coherent story — divergence between reported results and what the code actually reproduces is explicitly penalized.
- **Grading rubric (/20)**: EDA & preprocessing quality (3), modeling rigor/diversity (4), evaluation rigor — metrics, CV, no data leakage (4), interpretability & critical analysis (3), code quality/reproducibility (2), report writing (2), oral defense (2).
- **Required methodology** (independent of dataset choice): justify preprocessing choices, no data leakage (fit any transform/resampling only inside the train fold, before any other learning), compare **at least 3 model families** (linear/regularized, tree/ensemble, and one more), use a stratified CV protocol with justified hyperparameter search, evaluate on a held-out test set with metrics matched to the problem, and include an honest limitations/bias discussion (section 7 in the notebook) — this section is explicitly called out as the one that most differentiates grades.
- **AI tool usage policy**: generative AI assistance is allowed but must be disclosed in the report (which tool, for what task) and the student must be able to explain and justify *any* line in the submission, AI-assisted or not, in the oral defense. The README's "Utilisation d'outils d'IA" section is currently a placeholder ("À compléter avant la remise") — it must be filled in with concrete tool/task entries before submission. When assisting here, keep changes explainable and prefer the student's existing patterns over introducing unfamiliar ones, since they'll have to defend the code live.
- Academic integrity: no reuse of other students' or prior years' code/analysis without explicit citation.

## Repository layout

This repo is currently just a notebook + metadata — `data/`, `reports/figures/`, and `src/` referenced by the README and notebook **do not exist yet** and aren't tracked (see `.gitignore`); they are created locally when the notebook is run.

```
fraud-detection-ml/
├── notebooks/fraud_detection.ipynb   # the entire pipeline lives here
├── requirements.txt
└── README.md
```

- `data/raw/creditcard.csv` — must be downloaded manually (Kaggle credentials required), never committed.
- `reports/figures/` — where the notebook's `plt.savefig(...)` calls write PNGs; **create this directory before running the notebook** (`mkdir -p reports/figures`), matplotlib will not create it for you and every EDA/evaluation cell will throw `FileNotFoundError` otherwise.
- `src/` — mentioned in the README as a home for reusable functions "if extracted from the notebook," but nothing has been extracted yet; all logic currently lives inline in the notebook.

## Commands

There is no build/lint/test tooling — this is a single-notebook data science project. Setup and execution:

```bash
python -m venv .venv
source .venv/bin/activate           # .venv\Scripts\activate on Windows
pip install -r requirements.txt
mkdir -p data/raw reports/figures
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip   # requires ~/.kaggle/kaggle.json
jupyter lab notebooks/fraud_detection.ipynb
```

To validate the notebook runs end-to-end without opening Jupyter (useful after editing cells):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/fraud_detection.ipynb
```

There's no partial/single-cell CLI execution — the notebook must run top to bottom, since later sections depend on variables defined earlier (`df`, `X_train_scaled`, `best_estimators`, `results_df`, etc.). When editing one section, re-run from the top to confirm nothing downstream broke.

## Notebook architecture

`notebooks/fraud_detection.ipynb` is organized into 7 sequential sections, each building on state from the previous one — there's no modularization, so understanding data flow requires reading the whole notebook in order:

1. **EDA** — loads `data/raw/creditcard.csv`, checks nulls/duplicates, class balance (0.17% fraud), distributions of `Amount`/`Time`, correlation of each feature with `Class`.
2. **Preprocessing & split** — `train_test_split` (stratified on `Class`, `random_state=RANDOM_STATE`) happens **before** any scaling or resampling, to avoid leakage. Only `Time` and `Amount` are standardized (`V1`–`V28` are already PCA-whitened by the dataset's authors); the `StandardScaler` is fit on train only and applied to test.
3. **Modeling** — three `imblearn.pipeline.Pipeline` objects (Logistic Regression, Random Forest, k-NN), each with `SMOTE` as the first step, so oversampling happens fresh inside every CV fold rather than once on the whole train set — this is the leakage-prevention mechanism the grading rubric checks for.
4. **Model selection** — `RandomizedSearchCV` per model with `StratifiedKFold(n_splits=5)`, scored on `average_precision` (AUC-PR), since accuracy/ROC-AUC alone are misleading at this imbalance level.
5. **Evaluation** — final metrics on the held-out test set only (`classification_report`, confusion matrix, PR and ROC curves), plus explicit CV fold variance (mean ± std) as a stability check.
6. **Interpretability** — native `feature_importances_` for tree-based winners, else permutation importance (subsampled to 5000 test rows for speed) scored on `average_precision`; SHAP is mentioned as optional/not-yet-implemented.
7. **Critical discussion** — markdown-only, currently a placeholder for limitations/bias/next-steps writeup.

`RANDOM_STATE = 42` is set once in the first cell and threaded through every split/model/search — preserve this if adding new randomized steps, for reproducibility the report depends on.

Figures are saved as a side effect of EDA/evaluation cells directly to `reports/figures/*.png` at `dpi=150`; these are the figures the written report references, so renaming a model (the `models` dict keys) changes the corresponding `confusion_matrix_{name}.png` filename too.
