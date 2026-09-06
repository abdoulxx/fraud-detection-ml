---
title: "Détection de fraude par carte de crédit"
author: "Samb Abdoulaye Sidy"
date: "2026"
---

# Détection de fraude par carte de crédit

Sujet A — Projet d'examen Machine Learning — M2 Génie Informatique

Samb Abdoulaye Sidy — 2026

::: notes
Bonjour, je vais présenter mon projet de détection de fraude par carte de crédit. 15 minutes, je vais suivre la démarche du notebook du bout en bout.
:::

# Contexte et problématique

- 284 807 transactions bancaires européennes (sept. 2013), 492 fraudes (**0,17 %**)
- Variables `V1`-`V28` anonymisées par ACP, plus `Time` et `Amount`
- Objectif métier : minimiser à la fois les fraudes non détectées (coût direct) et les fausses alertes (coût opérationnel, expérience client)
- Déséquilibre extrême → l'exactitude est un indicateur trompeur (99,83 % en prédisant toujours "non-fraude")

::: notes
Le point clé à retenir dès le départ : ce déséquilibre conditionne tous les choix méthodologiques qui suivent — métriques, rééchantillonnage, protocole d'évaluation.
:::

# Analyse exploratoire des données

![Répartition des classes](figures/class_balance.png){width=45%}

- Aucune valeur manquante
- **1 081 doublons exacts détectés** — traité comme un risque de fuite de données, pas un simple nettoyage
- `Amount` fortement asymétrique ; `V14`, `V12`, `V10`, `V17` les plus corrélées à `Class`

::: notes
J'insiste sur les doublons : si on ne les retire pas avant le split, certaines lignes de test sont identiques à des lignes vues en entraînement — ça gonfle artificiellement les métriques. Je quantifierai l'effet mesuré plus loin.
:::

# Prétraitement — éviter toute fuite de données

Ordre strict des opérations :

1. Suppression des doublons (sur l'ensemble complet, avant tout split)
2. Split stratifié train/test 80/20 (`random_state=42`) — **avant** tout apprentissage statistique
3. `StandardScaler` sur `Time`/`Amount` uniquement, **fit sur train seul**
4. SMOTE intégré au pipeline `imblearn`, refit à l'intérieur de **chaque pli de CV**

::: notes
Le point 4 est le mécanisme central attendu par la consigne : SMOTE n'est jamais appliqué une seule fois sur tout le train, il est recalculé à chaque pli pour ne jamais voir les données de validation.
:::

# Modélisation — trois familles comparées

| Famille | Modèle | Rôle |
|---|---|---|
| Linéaire régularisé | Régression logistique | Baseline interprétable |
| Ensembliste (arbres) | Random Forest | Capture les interactions non linéaires |
| Instance-based | k-NN | Point de comparaison, non paramétrique |

Chaque modèle : pipeline `SMOTE → Classifieur`, protocole d'évaluation identique.

::: notes
Trois familles volontairement différentes dans leurs hypothèses, pour une comparaison honnête sur un protocole strictement identique.
:::

# Sélection de modèle

- `StratifiedKFold(n_splits=5)` — préserve la proportion de fraudes dans chaque pli
- `RandomizedSearchCV` par modèle, espaces de recherche justifiés (ex. Random Forest : `n_estimators`, `max_depth`, `min_samples_leaf`)
- Métrique de sélection : **AUC-PR** (aire sous précision-rappel), pas l'AUC-ROC seule ni l'exactitude

::: notes
Avec 0,17% de positifs, l'AUC-ROC reste artificiellement élevée même pour un modèle médiocre. L'AUC-PR est bien plus sensible à la vraie capacité à isoler les rares fraudes.
:::

# Résultats sur le jeu de test tenu à l'écart

![Courbes précision-rappel](figures/pr_curves.png){width=48%}

| Modèle | AUC-PR | AUC-ROC | Précision | Rappel |
|---|---|---|---|---|
| **Random Forest** | **0,818** | 0,960 | 89,2 % | 77,9 % |
| Régression logistique | 0,659 | 0,962 | 5,4 % | 87,4 % |
| k-NN | 0,540 | 0,926 | 31,9 % | 83,2 % |

::: notes
Random Forest domine largement sur l'AUC-PR, la métrique de sélection retenue, et offre le meilleur compromis précision/rappel au seuil par défaut.
:::

# Random Forest — modèle retenu

![Matrice de confusion — Random Forest](figures/confusion_matrix_Random_Forest.png){width=45%}

- 74 vrais positifs, 21 faux négatifs (fraudes manquées), 9 faux positifs, 56 642 vrais négatifs
- Variance CV entre plis : ± 0,032 — **la plus élevée des trois modèles**, une nuance assumée plutôt que masquée

::: notes
Je choisis Random Forest sur la performance moyenne, pas sur la stabilité — c'est un compromis honnête que je documente dans la section critique.
:::

# Interprétabilité

![Importance des variables](figures/feature_importance.png){width=45%}

- Importance native et importance par permutation convergent sur `V4`, `V10`, `V12`, `V14`
- SHAP envisagé mais non implémenté — limite assumée du travail

::: notes
La convergence des deux méthodes d'importance renforce la confiance dans le signal retenu par le modèle plutôt qu'un artefact de la méthode choisie.
:::

# Discussion critique et conclusion

- **Effet mesuré du dédoublonnage** : avant correction, AUC-PR = 0,873 (optimiste) ; après correction, 0,818 — preuve concrète du risque de fuite par doublons
- **Limites** : pas de calibrage de seuil de décision, fenêtre temporelle étroite (2 jours, 2013), anonymisation ACP limitant l'interprétation métier
- **Pistes** : calibrer le seuil sur un coût métier explicite, tester XGBoost, ajouter SHAP
- **IA générative** : Claude Code utilisé pour l'environnement, l'exécution de validation et un premier jet de rédaction — voir section dédiée du rapport

::: notes
Je termine sur la section que je considère la plus importante pour la notation : une analyse honnête de ce qui ne marche pas, plutôt que de ne présenter que des résultats favorables.
:::
