"""Génération et chargement des séries mensuelles de prix (FCFA/kg).

ATTENTION : les prix sont SIMULÉS (tendance + saisonnalité + bruit + un épisode de hausse).
Pour utiliser de vraies données, fournir un CSV avec les colonnes : date, produit, prix_fcfa_kg.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PRODUITS = {
    #         nom        prix de base  hausse/an  amplitude saison  mois du pic  bruit
    "mais":   ("Maïs",   220.0, 0.040, 0.18, 7, 0.035),
    "tomate": ("Tomate", 450.0, 0.030, 0.30, 4, 0.060),
    "igname": ("Igname", 330.0, 0.035, 0.14, 8, 0.030),
}
CHEMIN_DEFAUT = Path("data/prix_mensuels.csv")


def generer(debut: str = "2016-01", fin: str = "2026-10", graine: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(graine)
    dates = pd.date_range(debut, fin, freq="MS")
    t = np.arange(len(dates))
    lignes = []
    for cle, (nom, base, hausse, amp, pic, bruit) in PRODUITS.items():
        saison = 1 + amp * np.cos(2 * np.pi * (dates.month.to_numpy() - pic) / 12)
        ar = np.zeros(len(dates))                       # bruit autocorrélé, plus réaliste qu'un bruit blanc
        for i in range(1, len(dates)):
            ar[i] = 0.6 * ar[i - 1] + rng.normal(0, bruit)
        choc = np.where((dates >= "2022-03-01") & (dates <= "2022-12-01"), 0.12, 0.0)   # épisode de hausse
        prix = base * np.exp(hausse * t / 12) * saison * np.exp(ar + choc)
        lignes.append(pd.DataFrame({"date": dates, "produit": cle, "prix_fcfa_kg": prix.round(1)}))
    return pd.concat(lignes, ignore_index=True)


def sauvegarder(df: pd.DataFrame, chemin: Path = CHEMIN_DEFAUT) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(chemin, index=False)


def charger(chemin: Path = CHEMIN_DEFAUT) -> pd.DataFrame:
    df = pd.read_csv(chemin, parse_dates=["date"])
    manquantes = {"date", "produit", "prix_fcfa_kg"} - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans {chemin} : {sorted(manquantes)}")
    return df


def serie(df: pd.DataFrame, produit: str) -> pd.Series:
    s = df[df["produit"] == produit].set_index("date")["prix_fcfa_kg"].sort_index().asfreq("MS")
    if s.isna().any():
        s = s.interpolate()                              # comble les rares mois manquants
    return s.astype(float)
