# -*- coding: utf-8 -*-
"""
Tests pour EGARCH, GJR-GARCH et comparaison de modèles.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.garch_model import GARCHModel, EGARCHModel, GJRGARCHModel, compare_garch_models


def generate_asymmetric_returns(n=2000, seed=42):
    """
    Génère des rendements synthétiques avec effet de levier.
    Les rendements négatifs augmentent davantage la volatilité future.
    """
    np.random.seed(seed)
    returns = np.zeros(n)
    variance = np.zeros(n)
    omega, alpha, beta, gamma = 0.00001, 0.05, 0.85, 0.10

    variance[0] = omega / (1 - alpha - beta - gamma / 2)

    for t in range(1, n):
        indicator = 1.0 if returns[t - 1] < 0 else 0.0
        variance[t] = (omega + alpha * returns[t - 1] ** 2
                        + gamma * indicator * returns[t - 1] ** 2
                        + beta * variance[t - 1])
        returns[t] = np.sqrt(max(variance[t], 1e-12)) * np.random.standard_normal()

    return returns


# ============================================================
# EGARCH Tests
# ============================================================

class TestEGARCH:

    def setup_method(self):
        self.returns = generate_asymmetric_returns(n=2000, seed=42)

    def test_fit_converges(self):
        """EGARCH fit doit converger."""
        model = EGARCHModel(self.returns)
        result = model.fit()
        assert result['converged'], "EGARCH n'a pas convergé"

    def test_parameters_exist(self):
        """Les paramètres omega, alpha, beta, gamma doivent exister après fit."""
        model = EGARCHModel(self.returns)
        model.fit()
        assert model.omega is not None
        assert model.alpha is not None
        assert model.beta is not None
        assert model.gamma is not None

    def test_leverage_effect(self):
        """
        gamma < 0 pour EGARCH sur données avec effet de levier.
        (Convention EGARCH: gamma négatif = chocs négatifs augmentent la vol.)
        """
        model = EGARCHModel(self.returns)
        model.fit()
        assert model.gamma < 0, (
            f"gamma devrait être négatif (effet de levier), obtenu: {model.gamma:.4f}"
        )

    def test_log_likelihood_vs_garch(self):
        """EGARCH doit avoir une meilleure log-likelihood sur données asymétriques."""
        garch = GARCHModel(self.returns)
        garch_result = garch.fit()

        egarch = EGARCHModel(self.returns)
        egarch_result = egarch.fit()

        assert egarch_result['log_likelihood'] >= garch_result['log_likelihood'] - 10, (
            f"EGARCH LL ({egarch_result['log_likelihood']:.2f}) "
            f"beaucoup plus faible que GARCH LL ({garch_result['log_likelihood']:.2f})"
        )

    def test_forecast_positive(self):
        """Les prévisions de volatilité doivent être positives."""
        model = EGARCHModel(self.returns)
        model.fit()
        forecasts = model.forecast(horizon=10)
        assert len(forecasts) == 10
        assert np.all(forecasts > 0), "Les prévisions doivent être positives"

    def test_volatility_series_length(self):
        """La série de volatilité doit avoir la même longueur que les rendements."""
        model = EGARCHModel(self.returns)
        model.fit()
        vol = model.get_volatility_series()
        assert len(vol) == len(self.returns)

    def test_forecast_before_fit_raises(self):
        """Appeler forecast() avant fit() doit lever ValueError."""
        model = EGARCHModel(self.returns)
        with pytest.raises(ValueError):
            model.forecast(horizon=5)


# ============================================================
# GJR-GARCH Tests
# ============================================================

class TestGJRGARCH:

    def setup_method(self):
        self.returns = generate_asymmetric_returns(n=2000, seed=42)

    def test_fit_converges(self):
        """GJR-GARCH fit doit converger."""
        model = GJRGARCHModel(self.returns)
        result = model.fit()
        assert result['converged'], "GJR-GARCH n'a pas convergé"

    def test_constraints_satisfied(self):
        """Contraintes: omega > 0, alpha >= 0, beta >= 0, gamma >= 0."""
        model = GJRGARCHModel(self.returns)
        model.fit()
        assert model.omega > 0, f"omega doit être > 0, obtenu: {model.omega}"
        assert model.alpha >= 0, f"alpha doit être >= 0, obtenu: {model.alpha}"
        assert model.beta >= 0, f"beta doit être >= 0, obtenu: {model.beta}"
        assert model.gamma >= 0, f"gamma doit être >= 0, obtenu: {model.gamma}"

    def test_stationarity(self):
        """Condition de stationnarité: alpha + beta + gamma/2 < 1."""
        model = GJRGARCHModel(self.returns)
        model.fit()
        persistence = model.alpha + model.beta + model.gamma / 2
        assert persistence < 1, (
            f"Condition de stationnarité violée: "
            f"alpha + beta + gamma/2 = {persistence:.4f} >= 1"
        )

    def test_gamma_positive_leverage(self):
        """gamma > 0 sur données avec effet de levier (GJR convention)."""
        model = GJRGARCHModel(self.returns)
        model.fit()
        assert model.gamma > 0, (
            f"gamma devrait être > 0 (effet de levier), obtenu: {model.gamma:.4f}"
        )

    def test_forecast_positive(self):
        """Les prévisions doivent être positives."""
        model = GJRGARCHModel(self.returns)
        model.fit()
        forecasts = model.forecast(horizon=10)
        assert len(forecasts) == 10
        assert np.all(forecasts > 0), "Les prévisions doivent être positives"

    def test_volatility_series_length(self):
        """Série de volatilité = même longueur que rendements."""
        model = GJRGARCHModel(self.returns)
        model.fit()
        vol = model.get_volatility_series()
        assert len(vol) == len(self.returns)

    def test_forecast_before_fit_raises(self):
        """Appeler forecast() avant fit() doit lever ValueError."""
        model = GJRGARCHModel(self.returns)
        with pytest.raises(ValueError):
            model.forecast(horizon=5)


# ============================================================
# Model Comparison Tests
# ============================================================

class TestModelComparison:

    def setup_method(self):
        self.returns = generate_asymmetric_returns(n=2000, seed=42)

    def test_compare_returns_all_models(self):
        """compare_garch_models doit retourner les 3 modèles."""
        results = compare_garch_models(self.returns)
        assert 'GARCH' in results
        assert 'EGARCH' in results
        assert 'GJR-GARCH' in results

    def test_aic_bic_formula(self):
        """Vérifier que AIC et BIC sont calculés correctement."""
        results = compare_garch_models(self.returns)
        n = len(self.returns)

        for name, res in results.items():
            ll = res['log_likelihood']
            k = res['n_params']
            expected_aic = -2 * ll + 2 * k
            expected_bic = -2 * ll + k * np.log(n)
            assert abs(res['AIC'] - expected_aic) < 1e-6, (
                f"{name}: AIC incorrect ({res['AIC']} vs {expected_aic})"
            )
            assert abs(res['BIC'] - expected_bic) < 1e-6, (
                f"{name}: BIC incorrect ({res['BIC']} vs {expected_bic})"
            )

    def test_all_models_converge(self):
        """Les 3 modèles doivent converger."""
        results = compare_garch_models(self.returns)
        for name, res in results.items():
            assert res['converged'], f"{name} n'a pas convergé"

    def test_persistence_values(self):
        """La persistance doit être entre 0 et 1 pour tous les modèles."""
        results = compare_garch_models(self.returns)
        for name, res in results.items():
            assert 0 < res['persistence'] < 1, (
                f"{name}: persistance hors bornes ({res['persistence']:.4f})"
            )
