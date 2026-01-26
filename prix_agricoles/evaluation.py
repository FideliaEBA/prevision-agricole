"""Évaluation honnête par « backtest » : on simule des prévisions réelles dans le passé.

À chaque date d'origine, le modèle ne voit que l'historique antérieur et prévoit les `horizon`
mois suivants ; on compare ensuite aux prix réellement observés.
"""
import numpy as np
import pandas as pd

from .modeles import MODELES


def backtest(serie: pd.Series, modele, n_test: int = 24, horizon: int = 6) -> dict:
    erreurs, relatives = [], []
    for origine in range(len(serie) - n_test, len(serie), horizon):
        fin = min(origine + horizon, len(serie))
        prevu = modele(serie.iloc[:origine], fin - origine)
        reel = serie.iloc[origine:fin].to_numpy()
        erreurs.extend(prevu - reel)
        relatives.extend(np.abs(prevu - reel) / reel)
    erreurs = np.array(erreurs)
    return {"MAE": float(np.abs(erreurs).mean()), "RMSE": float(np.sqrt((erreurs ** 2).mean())),
            "MAPE_%": float(np.mean(relatives) * 100)}


def comparer(serie: pd.Series, n_test: int = 24, horizon: int = 6) -> pd.DataFrame:
    lignes = {nom: backtest(serie, f, n_test, horizon) for nom, f in MODELES.items()}
    return pd.DataFrame(lignes).T.round(2).sort_values("MAE")
