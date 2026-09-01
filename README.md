# Détection de fraude par carte de crédit

Projet d'examen — Machine Learning — M2 Génie Informatique (session 2026–2027).
Sujet A : détection de transactions bancaires frauduleuses sur un jeu de données extrêmement déséquilibré.

## Jeu de données

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle / Université Libre de Bruxelles).
284 807 transactions, 492 frauduleuses (0,17 %). Variables `V1`–`V28` (ACP anonymisée), `Time`, `Amount`, cible `Class`.

Le fichier `creditcard.csv` n'est pas versionné (voir `.gitignore`). Pour le récupérer :

1. Créer un compte Kaggle et une clé API (`kaggle.json`).
2. `kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip`
3. Le fichier attendu est `data/raw/creditcard.csv`.

## Structure du projet

```
fraud-detection-ml/
├── data/
│   ├── raw/            # creditcard.csv (non versionné)
│   └── processed/      # jeux de données transformés (non versionnés)
├── notebooks/
│   └── fraud_detection.ipynb   # pipeline complet EDA -> évaluation -> interprétabilité
├── reports/
│   └── figures/        # graphiques exportés pour le rapport écrit
├── src/                # fonctions réutilisables (si extraites du notebook)
├── requirements.txt
└── README.md
```

## Reproduire les résultats

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter lab notebooks/fraud_detection.ipynb
```

Exécuter le notebook de haut en bas. Toutes les figures et métriques présentées dans le rapport écrit sont générées par ce notebook.

## Utilisation d'outils d'IA

À compléter avant la remise (section obligatoire du rapport) : lister précisément les outils utilisés et pour quelles tâches.
