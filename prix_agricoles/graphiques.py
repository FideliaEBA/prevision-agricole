"""Graphique de prévision : historique récent, prévision et marge indicative."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def tracer_prevision(serie: pd.Series, prevision: pd.Series, nom: str, mape: float, chemin: Path) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    marge = mape / 100
    fig, ax = plt.subplots(figsize=(9, 4.8))
    recent = serie.iloc[-36:]
    ax.plot(recent.index, recent.values, color="#1f6f4a", lw=2, label="Prix observés")
    ax.plot(prevision.index, prevision.values, color="#2f6fde", lw=2, marker="o", label="Prévision")
    ax.fill_between(prevision.index, prevision.values * (1 - marge), prevision.values * (1 + marge),
                    color="#2f6fde", alpha=0.18, label=f"Marge indicative (± {mape:.0f} %, erreur moyenne du backtest)")
    ax.set_title(f"{nom} : prévision du prix mensuel (données simulées)")
    ax.set_ylabel("FCFA par kg")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
