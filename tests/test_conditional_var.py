# -*- coding: utf-8 -*-
"""
Tests pour Conditional VaR (GARCH-based) et extensions de backtesting.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conditional_var import ConditionalVaR
from src.backtesting import (
    exception_clustering,
    lookback_sensitivity,
    confidence_sensitivity,
)


def generate_regime_returns(seed=42):
    """
    Génère des rendements avec 3 régimes pour tester l'adaptativité.
    - Période calme (faible vol)
    - Période volatile (forte vol)
    - Période de récupération (vol moyenne)
    """
    np.random.seed(seed)
    returns_calm = np.random.normal(0, 0.01, 500)
    returns_volatile = np.random.normal(-0.005, 0.03, 200)
    returns_recovery = np.random.normal(0.001, 0.015, 300)
    return np.concatenate([returns_calm, returns_volatile, returns_recovery])


# ============================================================
# ConditionalVaR Tests
# ============================================================

class TestConditionalVaR:

    def setup_method(self):
        self.returns = generate_regime_returns(seed=42)
        self.cvar = ConditionalVaR(self.returns, portfolio_value=1_000_000)

    def test_garch_var_positive(self):
        """GARCH VaR doit retourner une valeur positive."""
        result = self.cvar.garch_var(confidence=0.95, horizon=1, garch_type='garch')
        assert result['var_pct'] > 0, f"VaR pct négatif: {result['var_pct']}"
        assert result['var_value'] > 0, f"VaR value négatif: {result['var_value']}"

    def test_garch_var_95_and_99(self):
        """VaR 99% > VaR 95%."""
        var_95 = self.cvar.garch_var(confidence=0.95)
        var_99 = self.cvar.garch_var(confidence=0.99)
        assert var_99['var_pct'] > var_95['var_pct'], (
            f"VaR 99% ({var_99['var_pct']:.4f}) <= VaR 95% ({var_95['var_pct']:.4f})"
        )

    def test_garch_var_returns_expected_keys(self):
        """Le résultat doit contenir les clés attendues."""
        result = self.cvar.garch_var(confidence=0.95)
        expected_keys = {'var_pct', 'var_value', 'sigma_current', 'sigma_forecast',
                         'model_params'}
        assert expected_keys.issubset(result.keys()), (
            f"Clés manquantes: {expected_keys - result.keys()}"
        )

    def test_ewma_var_positive(self):
        """EWMA VaR doit retourner une valeur positive."""
        result = self.cvar.ewma_var(confidence=0.95)
        assert result['var_pct'] > 0
        assert result['var_value'] > 0

    def test_ewma_var_reasonable(self):
        """EWMA VaR doit rester dans un intervalle raisonnable."""
        result = self.cvar.ewma_var(confidence=0.95)
        # Pour des rendements typiques, VaR 95% entre 0.1% et 20%
        assert 0.001 < result['var_pct'] < 0.20, (
            f"VaR hors bornes raisonnables: {result['var_pct']:.4f}"
        )

    def test_rolling_var_comparison_columns(self):
        """rolling_var_comparison doit retourner les bonnes colonnes."""
        df = self.cvar.rolling_var_comparison(
            window=250, confidence=0.95, methods=['static', 'ewma', 'garch']
        )
        expected_cols = {'return', 'var_static', 'var_ewma', 'var_garch',
                         'exception_static', 'exception_ewma', 'exception_garch'}
        assert expected_cols.issubset(df.columns), (
            f"Colonnes manquantes: {expected_cols - set(df.columns)}"
        )

    def test_rolling_var_comparison_length(self):
        """rolling_var_comparison doit retourner len(returns) - window lignes."""
        window = 250
        df = self.cvar.rolling_var_comparison(window=window, confidence=0.95)
        expected_len = len(self.returns) - window
        assert len(df) == expected_len, (
            f"Longueur incorrecte: {len(df)} vs {expected_len}"
        )

    def test_garch_var_adapts_to_regime(self):
        """
        GARCH VaR doit s'adapter: plus élevé en période volatile.
        On compare la volatilité estimée à la fin de la période calme
        vs à la fin de la période volatile.
        """
        # Période calme seulement
        calm_returns = self.returns[:500]
        cvar_calm = ConditionalVaR(calm_returns, portfolio_value=1_000_000)
        var_calm = cvar_calm.garch_var(confidence=0.95)

        # Inclut la période volatile
        volatile_returns = self.returns[:700]
        cvar_volatile = ConditionalVaR(volatile_returns, portfolio_value=1_000_000)
        var_volatile = cvar_volatile.garch_var(confidence=0.95)

        assert var_volatile['sigma_current'] > var_calm['sigma_current'], (
            f"Sigma devrait augmenter après la période volatile: "
            f"calm={var_calm['sigma_current']:.6f}, "
            f"volatile={var_volatile['sigma_current']:.6f}"
        )

    def test_compare_var_models_structure(self):
        """compare_var_models doit retourner les infos pour chaque méthode."""
        result = self.cvar.compare_var_models(window=250, confidence=0.95)
        assert 'static' in result
        assert 'ewma' in result
        assert 'garch' in result
        for method in ['static', 'ewma', 'garch']:
            assert 'exception_rate' in result[method]
            assert 'kupiec' in result[method]

    def test_garch_type_egarch(self):
        """Le paramètre garch_type='egarch' doit fonctionner."""
        result = self.cvar.garch_var(confidence=0.95, garch_type='egarch')
        assert result['var_pct'] > 0

    def test_garch_type_gjr(self):
        """Le paramètre garch_type='gjr' doit fonctionner."""
        result = self.cvar.garch_var(confidence=0.95, garch_type='gjr')
        assert result['var_pct'] > 0


# ============================================================
# Backtesting Enhancement Tests
# ============================================================

class TestExceptionClustering:

    def test_clustered_exceptions(self):
        """Exceptions groupées doivent montrer du clustering."""
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0, 0.01, n)
        var_estimates = np.ones(n) * 0.02

        # Créer un cluster d'exceptions (jours 100-110)
        returns[100:111] = -0.03

        result = exception_clustering(returns, var_estimates, confidence=0.95)

        assert result['max_consecutive'] >= 5, (
            f"max_consecutive devrait être >= 5, obtenu: {result['max_consecutive']}"
        )
        assert result['cluster_count'] >= 1

    def test_spread_exceptions(self):
        """Exceptions uniformément réparties: pas de clustering significatif."""
        np.random.seed(42)
        n = 500
        returns = np.zeros(n)
        var_estimates = np.ones(n) * 0.02

        # Exceptions espacées régulièrement (tous les 50 jours)
        for i in range(0, n, 50):
            returns[i] = -0.03

        result = exception_clustering(returns, var_estimates, confidence=0.95)

        assert result['max_consecutive'] == 1, (
            f"max_consecutive devrait être 1 (pas de clustering), "
            f"obtenu: {result['max_consecutive']}"
        )

    def test_exception_clustering_structure(self):
        """Le résultat doit contenir les clés attendues."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 500)
        var_estimates = np.ones(500) * 0.03

        result = exception_clustering(returns, var_estimates)
        expected_keys = {'exception_dates', 'max_consecutive', 'cluster_count',
                         'average_cluster_size', 'independence_test', 'interpretation'}
        assert expected_keys.issubset(result.keys()), (
            f"Clés manquantes: {expected_keys - result.keys()}"
        )


class TestLookbackSensitivity:

    def setup_method(self):
        np.random.seed(42)
        self.returns = np.random.normal(0, 0.02, 1500)

    def test_returns_correct_structure(self):
        """lookback_sensitivity doit retourner un DataFrame avec les bonnes colonnes."""
        result = lookback_sensitivity(self.returns, confidence=0.95,
                                       lookback_periods=[125, 250, 500])
        assert 'lookback' in result.columns
        assert 'var_95' in result.columns
        assert 'exception_rate' in result.columns
        assert len(result) == 3

    def test_shorter_lookback_more_variable(self):
        """
        Des lookback plus courts devraient produire des VaR plus variables.
        On vérifie que la VaR change avec le lookback (pas identique partout).
        """
        result = lookback_sensitivity(self.returns, confidence=0.95,
                                       lookback_periods=[125, 250, 500, 1000])
        var_values = result['var_95'].values
        # Les valeurs ne doivent pas toutes être identiques
        assert np.std(var_values) > 0, "Les VaR devraient varier selon le lookback"


class TestConfidenceSensitivity:

    def test_returns_correct_structure(self):
        """confidence_sensitivity doit retourner un dict par niveau."""
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0, 0.02, n)

        var_by_conf = {}
        for conf in [0.90, 0.95, 0.99]:
            var_by_conf[conf] = np.ones(n) * (0.02 + (conf - 0.90) * 0.1)

        result = confidence_sensitivity(returns, var_by_conf,
                                         confidence_levels=[0.90, 0.95, 0.99])
        assert 0.90 in result
        assert 0.95 in result
        assert 0.99 in result

    def test_higher_confidence_fewer_exceptions(self):
        """Des niveaux de confiance plus élevés => moins d'exceptions attendues."""
        np.random.seed(42)
        n = 1000
        returns = np.random.normal(0, 0.02, n)

        # VaR croissant avec la confiance
        var_by_conf = {
            0.90: np.ones(n) * 0.025,
            0.95: np.ones(n) * 0.035,
            0.99: np.ones(n) * 0.050,
        }

        result = confidence_sensitivity(returns, var_by_conf,
                                         confidence_levels=[0.90, 0.95, 0.99])

        rate_90 = result[0.90]['exception_rate']
        rate_99 = result[0.99]['exception_rate']
        assert rate_90 >= rate_99, (
            f"Le taux d'exceptions à 90% ({rate_90:.4f}) devrait être >= "
            f"celui à 99% ({rate_99:.4f})"
        )
