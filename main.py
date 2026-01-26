"""Prévision des prix agricoles : compare 4 modèles par backtest, puis prévoit les prochains mois.

Utilisation :
    python main.py                       # les 3 produits, 6 mois
    python main.py --produit tomate --horizon 3
    python main.py --donnees mes_prix.csv   # avec de vraies données (date, produit, prix_fcfa_kg)
"""
import argparse
from pathlib import Path

import pandas as pd

from prix_agricoles import donnees, evaluation, graphiques
from prix_agricoles.modeles import MODELES


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--produit", default="tous", choices=["tous", *donnees.PRODUITS])
    p.add_argument("--horizon", type=int, default=6, help="nombre de mois à prévoir")
    p.add_argument("--donnees", type=Path, default=donnees.CHEMIN_DEFAUT, help="fichier CSV des prix")
    a = p.parse_args()

    if not a.donnees.exists():
        print(f"{a.donnees} introuvable : génération de données simulées.")
        donnees.sauvegarder(donnees.generer(), a.donnees)
    df = donnees.charger(a.donnees)
    produits = list(donnees.PRODUITS) if a.produit == "tous" else [a.produit]
    Path("results").mkdir(exist_ok=True)
    rapport = []

    for cle in produits:
        nom = donnees.PRODUITS.get(cle, (cle,))[0]
        s = donnees.serie(df, cle)
        tableau = evaluation.comparer(s, n_test=24, horizon=6)
        meilleur = tableau.index[0]
        prevu = MODELES[meilleur](s, a.horizon)
        dates = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(), periods=a.horizon, freq="MS")
        prevision = pd.Series(prevu, index=dates)

        texte = [f"=== {nom} ===", "Comparaison des modèles (backtest sur 24 mois, prévisions à 6 mois) :",
                 tableau.to_string(), f"\nMeilleur modèle : {meilleur}", f"Prévision {nom} (FCFA/kg) :"]
        texte += [f"  {d:%Y-%m} : {v:7.1f}" for d, v in prevision.items()]
        print("\n".join(texte) + "\n")
        rapport.append("\n".join(texte))
        tableau.to_csv(f"results/comparaison_{cle}.csv", encoding="utf-8")
        graphiques.tracer_prevision(s, prevision, nom, tableau.loc[meilleur, "MAPE_%"], Path(f"results/prevision_{cle}.png"))

    Path("results/rapport.txt").write_text("\n\n".join(rapport) + "\n", encoding="utf-8")
    print("Résultats enregistrés dans le dossier results/")


if __name__ == "__main__":
    main()
