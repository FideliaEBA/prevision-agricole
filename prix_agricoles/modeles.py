"""Modèles de prévision. Chaque modèle a la même signature :

    modele(historique: pd.Series, horizon: int) -> np.ndarray   (prix prévus, en FCFA/kg)

Deux références simples (naïf saisonnier, moyenne mobile) servent de points de comparaison :
un modèle d'apprentissage n'est utile que s'il les bat.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

MIN_HISTORIQUE = 24


def naif_saisonnier(historique: pd.Series, horizon: int) -> np.ndarray:
    """Le prix d'un mois = le prix du même mois l'année précédente."""
    valeurs = list(historique.to_numpy())
    for _ in range(horizon):
        valeurs.append(valeurs[-12])
    return np.array(valeurs[-horizon:])


def moyenne_mobile(historique: pd.Series, horizon: int) -> np.ndarray:
    """Le prix reste égal à la moyenne des 3 derniers mois."""
    return np.full(horizon, historique.iloc[-3:].mean())


def _features(logs: np.ndarray, mois: int, t: int) -> list:
    """Variables explicatives du mois t, calculées uniquement avec le passé (logs[:t])."""
    return [logs[t - 1] - logs[t - 2], logs[t - 2] - logs[t - 3], logs[t - 3] - logs[t - 4],
            logs[t - 1] - logs[t - 13],                     # évolution sur un an
            logs[t - 12] - logs[t - 13],                    # variation du même mois l'an dernier (saisonnalité)
            np.sin(2 * np.pi * mois / 12), np.cos(2 * np.pi * mois / 12)]


def _prevoir_apprentissage(creer_modele, historique: pd.Series, horizon: int) -> np.ndarray:
    """Apprend la variation mensuelle du prix (en logarithme), puis prévoit mois après mois."""
    if len(historique) < MIN_HISTORIQUE:
        raise ValueError(f"Il faut au moins {MIN_HISTORIQUE} mois d'historique.")
    logs = list(np.log(historique.to_numpy()))
    mois_debut = historique.index[0].month - 1
    X = [_features(np.array(logs), (mois_debut + t) % 12, t) for t in range(13, len(logs))]
    y = [logs[t] - logs[t - 1] for t in range(13, len(logs))]
    modele = creer_modele().fit(X, y)
    for _ in range(horizon):
        t = len(logs)
        variation = modele.predict([_features(np.array(logs), (mois_debut + t) % 12, t)])[0]
        logs.append(logs[-1] + variation)
    return np.exp(logs[-horizon:])


def regression_ridge(historique: pd.Series, horizon: int) -> np.ndarray:
    return _prevoir_apprentissage(lambda: Ridge(alpha=1.0), historique, horizon)


def foret_aleatoire(historique: pd.Series, horizon: int) -> np.ndarray:
    return _prevoir_apprentissage(
        lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42),
        historique, horizon)


MODELES = {
    "Naïf saisonnier": naif_saisonnier,
    "Moyenne mobile (3 mois)": moyenne_mobile,
    "Régression Ridge": regression_ridge,
    "Forêt aléatoire": foret_aleatoire,
}
