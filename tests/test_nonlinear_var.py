# -*- coding: utf-8 -*-
"""
Tests Unitaires pour le VaR Non-Linéaire (Delta-Gamma)
======================================================
Tests pour valider les méthodes de VaR non-linéaire :
    - Delta-Normal VaR (approximation linéaire)
    - Delta-Gamma VaR (Cornish-Fisher et moment-matching)
    - Full Revaluation Monte Carlo VaR
"""

import sys
import os
import numpy as np
import pytest
from math import exp, log, sqrt

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nonlinear_var import NonLinearVaR, example_portfolio


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def option_portfolio():
    """Portefeuille mixte : actions + puts protecteurs."""
    positions = [
        {
            'type': 'stock',
            'quantity': 100,
            'price': 100.0,
            'sigma': 0.25,
        },
        {
            'type': 'option',
            'quantity': 50,
            'S': 100.0,
            'K': 95.0,
            'T': 0.25,
            'r': 0.03,
            'sigma': 0.25,
            'option_type': 'put',
        },
    ]
    return NonLinearVaR(positions)


@pytest.fixture
def stock_only_portfolio():
    """Portefeuille composé uniquement d'actions (gamma = 0)."""
    positions = [
        {
            'type': 'stock',
            'quantity': 100,
            'price': 100.0,
            'sigma': 0.25,
        },
    ]
    return NonLinearVaR(positions)


@pytest.fixture
def pure_option_portfolio():
    """Portefeuille composé uniquement d'options (forte non-linéarité)."""
    positions = [
        {
            'type': 'option',
            'quantity': 100,
            'S': 100.0,
            'K': 100.0,
            'T': 0.5,
            'r': 0.05,
            'sigma': 0.20,
            'option_type': 'call',
        },
    ]
    return NonLinearVaR(positions)


# ================================================================
# Tests Black-Scholes helpers
# ================================================================

class TestBlackScholes:
    """Tests pour les fonctions auxiliaires Black-Scholes."""

    def test_bs_price_atm_call(self):
        """
        Prix BS d'un call ATM connu :
        S=100, K=100, T=1, r=5%, σ=20% → C ≈ 10.45
        """
        price = NonLinearVaR.bs_price(
            S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call'
        )
        assert abs(price - 10.45) < 0.1, \
            f"Prix BS call ATM attendu ≈ 10.45, obtenu {price:.4f}"

    def test_bs_price_put_call_parity(self):
        """
        Parité put-call : C - P = S - K·exp(-rT)
        """
        S, K, T, r, sigma = 100, 95, 0.25, 0.03, 0.25
        call = NonLinearVaR.bs_price(S, K, T, r, sigma, 'call')
        put = NonLinearVaR.bs_price(S, K, T, r, sigma, 'put')
        parity = S - K * exp(-r * T)

        assert abs((call - put) - parity) < 1e-10, \
            f"Parité put-call violée : C-P={call - put:.6f}, S-Ke^(-rT)={parity:.6f}"

    def test_bs_delta_call_range(self):
        """Delta d'un call doit être dans (0, 1)."""
        delta = NonLinearVaR.bs_delta(
            S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type='call'
        )
        assert 0 < delta < 1, f"Delta call hors bornes : {delta}"

    def test_bs_delta_put_range(self):
        """Delta d'un put doit être dans (-1, 0)."""
        delta = NonLinearVaR.bs_delta(
            S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type='put'
        )
        assert -1 < delta < 0, f"Delta put hors bornes : {delta}"

    def test_bs_delta_deep_itm_call(self):
        """Un call très in-the-money a un delta proche de 1."""
        delta = NonLinearVaR.bs_delta(
            S=200, K=100, T=0.5, r=0.05, sigma=0.20, option_type='call'
        )
        assert delta > 0.99, f"Delta deep ITM call attendu > 0.99, obtenu {delta}"

    def test_bs_gamma_positive(self):
        """Gamma est toujours positif (pour calls et puts)."""
        gamma_call = NonLinearVaR.bs_gamma(
            S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type='call'
        )
        gamma_put = NonLinearVaR.bs_gamma(
            S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type='put'
        )
        assert gamma_call > 0, f"Gamma call négatif : {gamma_call}"
        assert gamma_put > 0, f"Gamma put négatif : {gamma_put}"

    def test_bs_gamma_call_equals_put(self):
        """Gamma d'un call = gamma d'un put (même sous-jacent/strike)."""
        params = dict(S=100, K=100, T=0.5, r=0.05, sigma=0.20)
        gamma_call = NonLinearVaR.bs_gamma(**params, option_type='call')
        gamma_put = NonLinearVaR.bs_gamma(**params, option_type='put')
        assert abs(gamma_call - gamma_put) < 1e-12, \
            f"Gamma call ≠ gamma put : {gamma_call} vs {gamma_put}"


# ================================================================
# Tests Delta-Normal VaR
# ================================================================

class TestDeltaNormalVaR:
    """Tests pour le VaR delta-normal (approximation linéaire)."""

    def test_var_positive(self, option_portfolio):
        """Le VaR delta-normal doit être positif."""
        var = option_portfolio.delta_normal_var(confidence=0.95)
        assert var > 0, f"VaR delta-normal négatif : {var}"

    def test_var_99_greater_than_95(self, option_portfolio):
        """VaR(99%) > VaR(95%)."""
        var_95 = option_portfolio.delta_normal_var(confidence=0.95)
        var_99 = option_portfolio.delta_normal_var(confidence=0.99)
        assert var_99 > var_95, \
            f"VaR 99% ({var_99}) <= VaR 95% ({var_95})"

    def test_stock_only_consistency(self, stock_only_portfolio):
        """
        Pour un portefeuille 100% actions, le VaR delta-normal doit être
        cohérent avec un calcul paramétrique simple :
        VaR = quantity * price * sigma * z_α * sqrt(horizon/252)
        """
        from scipy.stats import norm
        var = stock_only_portfolio.delta_normal_var(confidence=0.95, horizon=1)
        z = norm.ppf(0.95)
        # stock delta = 1 par unité
        expected = 100 * 100.0 * 0.25 * z * sqrt(1 / 252)
        assert abs(var - expected) / expected < 0.01, \
            f"VaR delta-normal stock ({var:.2f}) ≠ attendu ({expected:.2f})"


# ================================================================
# Tests Delta-Gamma VaR
# ================================================================

class TestDeltaGammaVaR:
    """Tests pour le VaR delta-gamma."""

    def test_var_positive(self, option_portfolio):
        """Le VaR delta-gamma doit être positif."""
        var_cf = option_portfolio.delta_gamma_var(
            confidence=0.95, method='cornish-fisher'
        )
        var_mm = option_portfolio.delta_gamma_var(
            confidence=0.95, method='moment-matching'
        )
        assert var_cf > 0, f"VaR delta-gamma (CF) négatif : {var_cf}"
        assert var_mm > 0, f"VaR delta-gamma (MM) négatif : {var_mm}"

    def test_differs_from_delta_normal(self, pure_option_portfolio):
        """
        Pour un portefeuille composé uniquement d'options, le delta-gamma VaR
        doit significativement différer du delta-normal VaR (gamma dominant).
        On utilise le portefeuille d'options pures pour maximiser l'effet gamma.
        """
        var_dn = pure_option_portfolio.delta_normal_var(confidence=0.95)
        var_dg = pure_option_portfolio.delta_gamma_var(confidence=0.95)
        # La différence relative doit être au moins 1%
        assert abs(var_dg - var_dn) / var_dn > 0.01, \
            f"Delta-gamma ({var_dg:.2f}) trop proche de delta-normal ({var_dn:.2f})"

    def test_stock_only_approx_delta_normal(self, stock_only_portfolio):
        """
        Pour un portefeuille 100% actions (gamma = 0),
        delta-gamma VaR ≈ delta-normal VaR.
        """
        var_dn = stock_only_portfolio.delta_normal_var(confidence=0.95)
        var_dg = stock_only_portfolio.delta_gamma_var(confidence=0.95)
        assert abs(var_dg - var_dn) / var_dn < 0.05, \
            f"Stock only : DG ({var_dg:.2f}) trop différent de DN ({var_dn:.2f})"

    def test_cornish_fisher_vs_moment_matching(self, option_portfolio):
        """
        Les deux méthodes (Cornish-Fisher et moment-matching) doivent donner
        des résultats du même ordre de grandeur (< 30% d'écart).
        """
        var_cf = option_portfolio.delta_gamma_var(
            confidence=0.95, method='cornish-fisher'
        )
        var_mm = option_portfolio.delta_gamma_var(
            confidence=0.95, method='moment-matching'
        )
        rel_diff = abs(var_cf - var_mm) / max(var_cf, var_mm)
        assert rel_diff < 0.30, \
            f"CF ({var_cf:.2f}) vs MM ({var_mm:.2f}) : écart {rel_diff:.1%} > 30%"

    def test_var_99_greater_than_95(self, option_portfolio):
        """VaR(99%) > VaR(95%) pour delta-gamma."""
        var_95 = option_portfolio.delta_gamma_var(confidence=0.95)
        var_99 = option_portfolio.delta_gamma_var(confidence=0.99)
        assert var_99 > var_95, \
            f"DG VaR 99% ({var_99}) <= DG VaR 95% ({var_95})"


# ================================================================
# Tests Full Revaluation Monte Carlo VaR
# ================================================================

class TestFullRevaluationVaR:
    """Tests pour le VaR par réévaluation complète Monte Carlo."""

    def test_var_positive(self, option_portfolio):
        """Le VaR full revaluation doit être positif."""
        var = option_portfolio.full_revaluation_var(
            confidence=0.95, n_simulations=20000, seed=42
        )
        assert var > 0, f"VaR full reval négatif : {var}"

    def test_reproducibility(self, option_portfolio):
        """Même seed → même résultat."""
        var1 = option_portfolio.full_revaluation_var(
            confidence=0.95, n_simulations=20000, seed=42
        )
        var2 = option_portfolio.full_revaluation_var(
            confidence=0.95, n_simulations=20000, seed=42
        )
        assert var1 == var2, \
            f"Résultats non reproductibles : {var1} vs {var2}"

    def test_stock_only_approx_delta_normal(self, stock_only_portfolio):
        """
        Pour un portefeuille 100% actions (payoff linéaire),
        full revaluation ≈ delta-normal (tolérance 15%).
        """
        var_dn = stock_only_portfolio.delta_normal_var(confidence=0.95)
        var_fr = stock_only_portfolio.full_revaluation_var(
            confidence=0.95, n_simulations=50000, seed=42
        )
        rel_diff = abs(var_fr - var_dn) / var_dn
        assert rel_diff < 0.15, \
            f"Stock only : FR ({var_fr:.2f}) vs DN ({var_dn:.2f}), écart {rel_diff:.1%}"

    def test_option_portfolio_captures_nonlinearity(self, pure_option_portfolio):
        """
        Pour un portefeuille d'options pures, le full reval et le delta-gamma
        capturent tous deux la non-linéarité, donc doivent être plus proches
        entre eux qu'avec le delta-normal.
        """
        var_dn = pure_option_portfolio.delta_normal_var(confidence=0.95)
        var_dg = pure_option_portfolio.delta_gamma_var(confidence=0.95)
        var_fr = pure_option_portfolio.full_revaluation_var(
            confidence=0.95, n_simulations=50000, seed=42
        )
        # |FR - DG| < |FR - DN| (full reval plus proche de DG que de DN)
        diff_dg = abs(var_fr - var_dg)
        diff_dn = abs(var_fr - var_dn)
        assert diff_dg < diff_dn, \
            f"|FR-DG|={diff_dg:.2f} >= |FR-DN|={diff_dn:.2f}, " \
            f"la non-linéarité n'est pas mieux capturée par DG"

    def test_var_99_greater_than_95(self, option_portfolio):
        """VaR(99%) > VaR(95%) pour full revaluation."""
        var_95 = option_portfolio.full_revaluation_var(
            confidence=0.95, n_simulations=20000, seed=42
        )
        var_99 = option_portfolio.full_revaluation_var(
            confidence=0.99, n_simulations=20000, seed=42
        )
        assert var_99 > var_95, \
            f"FR VaR 99% ({var_99}) <= FR VaR 95% ({var_95})"


# ================================================================
# Tests Compare Methods
# ================================================================

class TestCompareMethods:
    """Tests pour la comparaison des trois méthodes."""

    def test_returns_all_values(self, option_portfolio):
        """compare_methods doit retourner les trois VaR et les écarts."""
        result = option_portfolio.compare_methods(confidence=0.95)
        assert 'var_delta_normal' in result
        assert 'var_delta_gamma' in result
        assert 'var_full_reval' in result
        assert 'diff_dg_vs_dn' in result
        assert 'diff_fr_vs_dn' in result

    def test_all_values_positive(self, option_portfolio):
        """Toutes les valeurs de VaR doivent être positives."""
        result = option_portfolio.compare_methods(confidence=0.95)
        assert result['var_delta_normal'] > 0
        assert result['var_delta_gamma'] > 0
        assert result['var_full_reval'] > 0

    def test_relative_differences_reported(self, option_portfolio):
        """Les écarts relatifs doivent être cohérents avec les VaR."""
        result = option_portfolio.compare_methods(confidence=0.95)
        # Vérifier la cohérence de diff_dg_vs_dn
        expected_diff = (
            (result['var_delta_gamma'] - result['var_delta_normal'])
            / result['var_delta_normal']
        )
        assert abs(result['diff_dg_vs_dn'] - expected_diff) < 1e-10


# ================================================================
# Tests Example Portfolio
# ================================================================

class TestExamplePortfolio:
    """Tests pour le portefeuille d'exemple."""

    def test_example_portfolio_runs(self):
        """Le portefeuille d'exemple doit être créé sans erreur."""
        nlv = example_portfolio()
        assert isinstance(nlv, NonLinearVaR)

    def test_example_portfolio_all_methods(self):
        """Toutes les méthodes doivent fonctionner sur l'exemple."""
        nlv = example_portfolio()
        var_dn = nlv.delta_normal_var(confidence=0.95)
        var_dg = nlv.delta_gamma_var(confidence=0.95)
        var_fr = nlv.full_revaluation_var(
            confidence=0.95, n_simulations=20000, seed=42
        )
        assert var_dn > 0
        assert var_dg > 0
        assert var_fr > 0
