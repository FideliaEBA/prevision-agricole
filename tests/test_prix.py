"""Tests : python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from prix_agricoles import donnees, evaluation
from prix_agricoles.modeles import MODELES, naif_saisonnier


class TestPrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = donnees.generer()
        cls.serie = donnees.serie(cls.df, "mais")

    def test_donnees_reproductibles_et_positives(self):
        self.assertTrue(donnees.generer().equals(self.df))
        self.assertTrue((self.df["prix_fcfa_kg"] > 0).all())

    def test_serie_mensuelle_sans_trou(self):
        self.assertEqual(self.serie.index.freqstr, "MS")
        self.assertFalse(self.serie.isna().any())

    def test_naif_saisonnier_repete_l_annee_precedente(self):
        prevu = naif_saisonnier(self.serie, 3)
        np.testing.assert_allclose(prevu, self.serie.iloc[-12:-9].to_numpy())

    def test_chaque_modele_renvoie_h_prix_positifs(self):
        for nom, modele in MODELES.items():
            prevu = modele(self.serie, 6)
            self.assertEqual(len(prevu), 6, nom)
            self.assertTrue((prevu > 0).all(), nom)

    def test_pas_de_fuite_du_futur(self):
        """Modifier les prix futurs ne doit pas changer la prévision faite avant."""
        modifiee = self.serie.copy()
        modifiee.iloc[-6:] *= 3
        for nom, modele in MODELES.items():
            np.testing.assert_allclose(modele(self.serie.iloc[:-6], 3), modele(modifiee.iloc[:-6], 3), err_msg=nom)

    def test_backtest_metriques(self):
        m = evaluation.backtest(self.serie, naif_saisonnier)
        self.assertTrue(all(v >= 0 for v in m.values()))


if __name__ == "__main__":
    unittest.main()
