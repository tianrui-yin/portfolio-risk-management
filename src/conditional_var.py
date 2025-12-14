# -*- coding: utf-8 -*-
"""
VaR Conditionnel (GARCH-based)
==============================
Calcul du VaR avec volatilité conditionnelle (time-varying σ).

Contrairement au VaR statique qui utilise une volatilité constante,
le VaR conditionnel adapte σ_t via les modèles GARCH, capturant
le clustering de volatilité et l'effet de levier.

Méthodes:
    - GARCH VaR: σ_t du GARCH(1,1), EGARCH ou GJR-GARCH
    - EWMA VaR: σ_t du modèle EWMA (RiskMetrics, λ=0.94)
    - Rolling comparison: comparaison glissante static vs EWMA vs GARCH
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .garch_model import GARCHModel, EGARCHModel, GJRGARCHModel, ewma_volatility
from .backtesting import kupiec_test, christoffersen_test


class ConditionalVaR:
    """
    VaR basé sur la volatilité conditionnelle GARCH.

    Attributs:
    ----------
    returns : np.array
        Série des rendements
    portfolio_value : float
        Valeur du portefeuille

    Exemple:
    --------
    >>> cvar = ConditionalVaR(returns, portfolio_value=1_000_000)
    >>> result = cvar.garch_var(confidence=0.95)
    >>> print(f"VaR GARCH: {result['var_pct']:.4%}")
    """

    def __init__(self, returns, portfolio_value=1_000_000):
        self.returns = np.array(returns)
        self.portfolio_value = portfolio_value

    def garch_var(self, confidence=0.95, horizon=1, garch_type='garch'):
        """
        Calcule le VaR basé sur la volatilité GARCH.

        1. Fit le modèle GARCH aux rendements
        2. Récupère σ_t (dernière volatilité conditionnelle)
        3. Prévoit σ pour les prochains jours
        4. VaR = z_α * σ_forecast * portfolio_value

        Paramètres:
        -----------
        confidence : float
            Niveau de confiance (ex: 0.95)
        horizon : int
            Horizon en jours
        garch_type : str
            Type de modèle: 'garch', 'egarch', 'gjr'

        Retourne:
        ---------
        dict : var_pct, var_value, sigma_current, sigma_forecast, model_params
        """
        # Sélection du modèle
        if garch_type == 'egarch':
            model = EGARCHModel(self.returns)
        elif garch_type == 'gjr':
            model = GJRGARCHModel(self.returns)
        else:
            model = GARCHModel(self.returns)

        fit_result = model.fit()

        # Dernière volatilité conditionnelle
        vol_series = model.get_volatility_series()
        sigma_current = vol_series[-1]

        # Prévision de volatilité
        forecast = model.forecast(horizon=horizon)
        # Pour le VaR multi-jours, utiliser la volatilité cumulée
        if horizon == 1:
            sigma_forecast = forecast[0]
        else:
            # σ_cumulée = sqrt(Σ σ²_h) pour h=1..horizon
            sigma_forecast = np.sqrt(np.sum(forecast ** 2))

        # Quantile normal
        z_alpha = norm.ppf(confidence)

        # VaR = z_α * σ_forecast
        var_pct = z_alpha * sigma_forecast
        var_value = var_pct * self.portfolio_value

        return {
            'var_pct': var_pct,
            'var_value': var_value,
            'sigma_current': sigma_current,
            'sigma_forecast': sigma_forecast,
            'model_params': fit_result,
        }

    def ewma_var(self, confidence=0.95, horizon=1, lambda_=0.94):
        """
        Calcule le VaR basé sur la volatilité EWMA (RiskMetrics).

        Paramètres:
        -----------
        confidence : float
            Niveau de confiance
        horizon : int
            Horizon en jours
        lambda_ : float
            Paramètre de lissage EWMA (défaut: 0.94)

        Retourne:
        ---------
        dict : var_pct, var_value, sigma_current
        """
        ewma_vol = ewma_volatility(self.returns, lambda_param=lambda_)

        sigma_current = ewma_vol[-1]
        sigma_forecast = sigma_current * np.sqrt(horizon)

        z_alpha = norm.ppf(confidence)
        var_pct = z_alpha * sigma_forecast
        var_value = var_pct * self.portfolio_value

        return {
            'var_pct': var_pct,
            'var_value': var_value,
            'sigma_current': sigma_current,
        }

    def rolling_var_comparison(self, window=250, confidence=0.95,
                                methods=None):
        """
        Comparaison glissante du VaR: static vs EWMA vs GARCH.

        Pour chaque jour t > window:
            - Static: σ constant = std des 'window' derniers jours
            - EWMA: σ_t EWMA sur l'historique complet jusqu'à t
            - GARCH: σ_t du GARCH(1,1) fit sur les 'window' derniers jours

        Paramètres:
        -----------
        window : int
            Taille de la fenêtre glissante
        confidence : float
            Niveau de confiance
        methods : list
            Méthodes à comparer (défaut: ['static', 'ewma', 'garch'])

        Retourne:
        ---------
        pd.DataFrame : colonnes return, var_static, var_ewma, var_garch,
                        exception_static, exception_ewma, exception_garch
        """
        if methods is None:
            methods = ['static', 'ewma', 'garch']

        n = len(self.returns)
        n_out = n - window
        z_alpha = norm.ppf(confidence)

        results = {
            'return': self.returns[window:],
        }

        # Static VaR: rolling std
        if 'static' in methods:
            var_static = np.zeros(n_out)
            for t in range(n_out):
                sigma = np.std(self.returns[t:t + window], ddof=1)
                var_static[t] = z_alpha * sigma
            results['var_static'] = var_static
            results['exception_static'] = (-self.returns[window:] > var_static).astype(int)

        # EWMA VaR
        if 'ewma' in methods:
            ewma_vol = ewma_volatility(self.returns, lambda_param=0.94)
            # Utiliser la vol EWMA au jour t-1 pour prédire le VaR du jour t
            var_ewma = z_alpha * ewma_vol[window - 1:-1]
            results['var_ewma'] = var_ewma
            results['exception_ewma'] = (-self.returns[window:] > var_ewma).astype(int)

        # GARCH VaR: fit une seule fois sur tout l'échantillon,
        # puis utiliser la série de vol conditionnelle
        if 'garch' in methods:
            garch = GARCHModel(self.returns)
            garch.fit()
            garch_vol = garch.get_volatility_series()
            # Utiliser la vol conditionnelle du jour t-1 pour le VaR du jour t
            var_garch = z_alpha * garch_vol[window - 1:-1]
            results['var_garch'] = var_garch
            results['exception_garch'] = (-self.returns[window:] > var_garch).astype(int)

        return pd.DataFrame(results)

    def compare_var_models(self, window=250, confidence=0.95):
        """
        Comparaison complète static vs EWMA vs GARCH VaR.

        Pour chaque méthode:
            - Taux d'exceptions
            - Test de Kupiec
            - Test de Christoffersen

        Paramètres:
        -----------
        window : int
            Fenêtre glissante
        confidence : float
            Niveau de confiance

        Retourne:
        ---------
        dict : résultats par méthode {'static': {...}, 'ewma': {...}, 'garch': {...}}
        """
        df = self.rolling_var_comparison(window=window, confidence=confidence)
        returns_test = df['return'].values

        results = {}
        for method in ['static', 'ewma', 'garch']:
            var_col = f'var_{method}'
            exc_col = f'exception_{method}'

            if var_col not in df.columns:
                continue

            var_estimates = df[var_col].values
            exceptions = df[exc_col].values

            exception_rate = np.mean(exceptions)
            kupiec = kupiec_test(returns_test, var_estimates, confidence)
            chris = christoffersen_test(returns_test, var_estimates, confidence)

            results[method] = {
                'exception_rate': exception_rate,
                'n_exceptions': int(np.sum(exceptions)),
                'expected_rate': 1 - confidence,
                'kupiec': kupiec,
                'christoffersen': chris,
            }

        return results
