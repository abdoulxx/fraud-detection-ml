# Détection de fraude par carte de crédit

Projet — Machine Learning.
Sujet : détection de transactions bancaires frauduleuses sur un jeu de données extrêmement déséquilibré.

## Jeu de données

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle / Université Libre de Bruxelles).
284 807 transactions, 492 frauduleuses (0,17 %). Variables `V1`–`V28` (ACP anonymisée), `Time`, `Amount`, cible `Class`.

Le fichier `creditcard.csv` n'est pas versionné. Pour le récupérer :

1. Créer un compte Kaggle.
2. Télécharger l'archive `.zip` du dataset depuis la [page Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) (bouton "Download").
3. Décompresser l'archive et placer le fichier `creditcard.csv` dans `data/raw/`.
4. Le fichier attendu est `data/raw/creditcard.csv`.

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

- **Outil** : Claude Code (assistant IA en ligne de commande, Anthropic), utilisé sur deux machines (Windows puis macOS) à différentes étapes du projet.
- **Tâches assistées** :
  - Mise en place initiale du projet (Windows) : structure des dossiers, `requirements.txt`, `.gitignore`, squelette du notebook et du README, création du venv et installation des dépendances.
  - Écriture du code des cellules du notebook (EDA, prétraitement, pipelines de modélisation, recherche d'hyperparamètres, évaluation, interprétabilité) à partir des consignes du sujet.
  - Diagnostic et correction de deux bugs réels rencontrés en cours de développement : une fuite de données par 1081 lignes dupliquées (correctif : suppression avant le split train/test) et un plantage machine dû à un parallélisme imbriqué (`n_jobs=-1` simultanément sur `RandomForestClassifier` et `RandomizedSearchCV`, corrigé en limitant le parallélisme à un seul niveau).
  - Sur macOS : diagnostic et résolution d'un problème de bibliothèque native manquante (`libomp`) bloquant l'import de XGBoost/SHAP.
  - Aide à la rédaction du rapport écrit, rédigé par Samb Abdoulaye Sidy avec l'assistance de Claude Code pour la formulation et la structuration.