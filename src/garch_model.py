# -*- coding: utf-8 -*-
"""
Modèles GARCH
=============
Estimation de la volatilité conditionnelle avec les modèles GARCH.

Référence: Hull, Chapitre 23 - Estimating Volatilities and Correlations

Modèles implémentés:
    1. GARCH(1,1):  σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
    2. EGARCH (Nelson, 1991):  ln(σ²_t) = ω + α * [|z_{t-1}| - E|z|] + γ * z_{t-1} + β * ln(σ²_{t-1})
    3. GJR-GARCH (Glosten-Jagannathan-Runkle, 1993):  σ²_t = ω + (α + γ * I_{t-1}) * r²_{t-1} + β * σ²_{t-1}

Les modèles EGARCH et GJR-GARCH capturent l'effet de levier:
    les rendements négatifs augmentent davantage la volatilité future.
"""

import numpy as np
from scipy.optimize import minimize


class GARCHModel:
    """
    Implémentation du modèle GARCH(1,1).

    Attributs:
    ----------
    returns : np.array
        Série des rendements
    omega : float
        Paramètre ω (terme constant)
    alpha : float
        Paramètre α (réaction aux chocs)
    beta : float
        Paramètre β (persistance)

    Exemple:
    --------
    >>> model = GARCHModel(returns)
    >>> model.fit()
    >>> vol_forecast = model.forecast(horizon=10)
    """

    def __init__(self, returns):
        """
        Initialise le modèle GARCH.

        Paramètres:
        -----------
        returns : array-like
            Rendements historiques
        """
        self.returns = np.array(returns)
        self.n_obs = len(returns)

        # Paramètres (à estimer)
        self.omega = None
        self.alpha = None
        self.beta = None

        # Volatilités conditionnelles estimées
        self.conditional_vol = None

    def _compute_variance(self, params):
        """
        Calcule les variances conditionnelles pour des paramètres donnés.

        Paramètres:
        -----------
        params : tuple
            (omega, alpha, beta)

        Retourne:
        ---------
        np.array : variances conditionnelles σ²_t
        """
        omega, alpha, beta = params
        n = self.n_obs
        variance = np.zeros(n)

        # Initialisation: variance inconditionnelle
        # E[σ²] = ω / (1 - α - β)
        if (1 - alpha - beta) > 0.001:
            var_uncond = omega / (1 - alpha - beta)
        else:
            var_uncond = np.var(self.returns)

        variance[0] = var_uncond

        # Récurrence GARCH(1,1)
        for t in range(1, n):
            variance[t] = omega + alpha * self.returns[t-1]**2 + beta * variance[t-1]

        return variance

    def _negative_log_likelihood(self, params):
        """
        Calcule la log-vraisemblance négative (à minimiser).

        Sous hypothèse de normalité conditionnelle:
            r_t | I_{t-1} ~ N(0, σ²_t)

        Log-vraisemblance:
            L = -0.5 * Σ [log(σ²_t) + r²_t / σ²_t]

        Paramètres:
        -----------
        params : tuple
            (omega, alpha, beta)

        Retourne:
        ---------
        float : -log(L)
        """
        omega, alpha, beta = params

        # Contraintes
        if omega <= 0 or alpha < 0 or beta < 0:
            return 1e10
        if alpha + beta >= 1:
            return 1e10

        variance = self._compute_variance(params)

        # Éviter log(0) et division par 0
        variance = np.maximum(variance, 1e-10)

        # Log-vraisemblance (sans constante)
        log_likelihood = -0.5 * np.sum(
            np.log(variance) + self.returns**2 / variance
        )

        return -log_likelihood  # On minimise le négatif

    def fit(self):
        """
        Estime les paramètres du modèle GARCH(1,1) par maximum de vraisemblance.

        Retourne:
        ---------
        dict : paramètres estimés et statistiques
        """
        # Variance empirique pour initialisation
        var_emp = np.var(self.returns)

        # Paramètres initiaux
        # Typiquement: α ≈ 0.1, β ≈ 0.85
        alpha_init = 0.10
        beta_init = 0.85
        omega_init = var_emp * (1 - alpha_init - beta_init)

        initial_params = [omega_init, alpha_init, beta_init]

        # Optimisation
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=[(1e-8, None), (0, 0.999), (0, 0.999)]
        )

        # Stocker les paramètres
        self.omega = result.x[0]
        self.alpha = result.x[1]
        self.beta = result.x[2]

        # Calculer les volatilités conditionnelles
        variance = self._compute_variance(result.x)
        self.conditional_vol = np.sqrt(variance)

        # Variance inconditionnelle (long-terme)
        if (1 - self.alpha - self.beta) > 0.001:
            long_term_var = self.omega / (1 - self.alpha - self.beta)
        else:
            long_term_var = var_emp

        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'persistence': self.alpha + self.beta,
            'long_term_vol': np.sqrt(long_term_var),
            'log_likelihood': -result.fun,
            'converged': result.success
        }

    def forecast(self, horizon=1):
        """
        Prévision de la volatilité future.

        Formule de prévision GARCH(1,1):
            E[σ²_{t+h}] = VL + (α + β)^h * (σ²_t - VL)

        où VL = ω / (1 - α - β) est la variance long-terme.

        Paramètres:
        -----------
        horizon : int
            Nombre de jours de prévision

        Retourne:
        ---------
        np.array : volatilités prévues
        """
        if self.omega is None:
            raise ValueError("Le modèle doit être estimé d'abord (appeler fit())")

        # Variance long-terme
        persistence = self.alpha + self.beta
        if (1 - persistence) > 0.001:
            var_long_term = self.omega / (1 - persistence)
        else:
            var_long_term = self.conditional_vol[-1]**2

        # Dernière variance observée
        var_current = self.conditional_vol[-1]**2

        # Prévisions
        forecasts = np.zeros(horizon)
        for h in range(horizon):
            forecasts[h] = var_long_term + (persistence**(h+1)) * (var_current - var_long_term)

        # Retourner en volatilité (écart-type)
        return np.sqrt(forecasts)

    def get_volatility_series(self):
        """
        Retourne la série complète des volatilités conditionnelles.

        Retourne:
        ---------
        np.array : volatilités conditionnelles estimées
        """
        if self.conditional_vol is None:
            raise ValueError("Le modèle doit être estimé d'abord")

        return self.conditional_vol


class EGARCHModel:
    """
    Modèle EGARCH (Nelson, 1991).

    ln(σ²_t) = ω + α * [|z_{t-1}| - E|z_{t-1}|] + γ * z_{t-1} + β * ln(σ²_{t-1})

    où z_t = r_t / σ_t (résidus standardisés), E|z| = sqrt(2/π) pour la loi normale.

    Avantages:
        - Pas de contraintes de positivité (le log garantit σ² > 0)
        - γ < 0 capture l'effet de levier (rendements négatifs → vol plus élevée)

    Paramètres: ω, α, β, γ
    """

    def __init__(self, returns):
        self.returns = np.array(returns)
        self.n_obs = len(returns)
        self.omega = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.conditional_vol = None
        self._log_likelihood = None

    def _compute_log_variance(self, params):
        """
        Calcule ln(σ²_t) via la récurrence EGARCH.

        Retourne:
        ---------
        np.array : log-variances conditionnelles ln(σ²_t)
        """
        omega, alpha, beta, gamma = params
        n = self.n_obs
        log_var = np.zeros(n)
        e_abs_z = np.sqrt(2.0 / np.pi)  # E[|z|] sous normalité

        # Initialisation: log de la variance empirique
        var_emp = np.var(self.returns)
        log_var[0] = np.log(max(var_emp, 1e-10))

        for t in range(1, n):
            sigma_prev = np.sqrt(max(np.exp(log_var[t - 1]), 1e-10))
            z_prev = self.returns[t - 1] / sigma_prev

            log_var[t] = (omega
                          + alpha * (np.abs(z_prev) - e_abs_z)
                          + gamma * z_prev
                          + beta * log_var[t - 1])

            # Borner pour éviter overflow/underflow
            log_var[t] = np.clip(log_var[t], -30, 10)

        return log_var

    def _negative_log_likelihood(self, params):
        """Log-vraisemblance négative pour EGARCH."""
        try:
            log_var = self._compute_log_variance(params)
            variance = np.exp(log_var)
            variance = np.maximum(variance, 1e-10)

            log_likelihood = -0.5 * np.sum(
                log_var + self.returns ** 2 / variance
            )

            if not np.isfinite(log_likelihood):
                return 1e10

            return -log_likelihood
        except (FloatingPointError, OverflowError):
            return 1e10

    def fit(self):
        """
        Estime les paramètres EGARCH par maximum de vraisemblance.
        """
        var_emp = np.var(self.returns)

        initial_params = [
            np.log(var_emp) * 0.05,  # omega
            0.15,                     # alpha
            0.95,                     # beta
            -0.05,                    # gamma (négatif = effet de levier)
        ]

        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=[
                (None, None),      # omega: pas de contrainte
                (None, None),      # alpha: pas de contrainte
                (-0.9999, 0.9999), # beta: |β| < 1 pour stationnarité
                (None, None),      # gamma: pas de contrainte
            ],
            options={'maxiter': 1000}
        )

        self.omega = result.x[0]
        self.alpha = result.x[1]
        self.beta = result.x[2]
        self.gamma = result.x[3]

        log_var = self._compute_log_variance(result.x)
        self.conditional_vol = np.sqrt(np.exp(log_var))
        self._log_likelihood = -result.fun

        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'persistence': np.abs(self.beta),
            'log_likelihood': self._log_likelihood,
            'converged': result.success,
        }

    def forecast(self, horizon=1):
        """
        Prévision de la volatilité future EGARCH.

        On utilise la récurrence en log-variance avec E[z_t] = 0
        et E[|z_t|] = sqrt(2/π) pour les termes futurs.
        """
        if self.omega is None:
            raise ValueError("Le modèle doit être estimé d'abord (appeler fit())")

        e_abs_z = np.sqrt(2.0 / np.pi)
        log_var_current = np.log(self.conditional_vol[-1] ** 2)

        forecasts = np.zeros(horizon)
        log_var_h = log_var_current

        for h in range(horizon):
            # E[z] = 0, E[|z|] = sqrt(2/π), donc:
            # E[ln(σ²_{t+h+1})] = ω + α*(E[|z|] - E[|z|]) + γ*0 + β*ln(σ²_{t+h})
            # = ω + β * ln(σ²_{t+h})
            log_var_h = self.omega + self.beta * log_var_h
            forecasts[h] = np.sqrt(np.exp(log_var_h))

        return forecasts

    def get_volatility_series(self):
        """Retourne la série des volatilités conditionnelles."""
        if self.conditional_vol is None:
            raise ValueError("Le modèle doit être estimé d'abord")
        return self.conditional_vol


class GJRGARCHModel:
    """
    Modèle GJR-GARCH (Glosten-Jagannathan-Runkle, 1993).

    σ²_t = ω + (α + γ * I_{t-1}) * r²_{t-1} + β * σ²_{t-1}

    où I_{t-1} = 1 si r_{t-1} < 0 (indicateur de rendement négatif).

    γ > 0 signifie que les rendements négatifs ont un impact plus fort
    sur la volatilité future (effet de levier).

    Condition de stationnarité: α + β + γ/2 < 1
    """

    def __init__(self, returns):
        self.returns = np.array(returns)
        self.n_obs = len(returns)
        self.omega = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.conditional_vol = None
        self._log_likelihood = None

    def _compute_variance(self, params):
        """
        Calcule les variances conditionnelles GJR-GARCH.
        """
        omega, alpha, beta, gamma = params
        n = self.n_obs
        variance = np.zeros(n)

        # Variance inconditionnelle: ω / (1 - α - β - γ/2)
        denom = 1 - alpha - beta - gamma / 2
        if denom > 0.001:
            var_uncond = omega / denom
        else:
            var_uncond = np.var(self.returns)

        variance[0] = var_uncond

        for t in range(1, n):
            indicator = 1.0 if self.returns[t - 1] < 0 else 0.0
            variance[t] = (omega
                           + (alpha + gamma * indicator) * self.returns[t - 1] ** 2
                           + beta * variance[t - 1])
            variance[t] = max(variance[t], 1e-10)

        return variance

    def _negative_log_likelihood(self, params):
        """Log-vraisemblance négative pour GJR-GARCH."""
        omega, alpha, beta, gamma = params

        # Contraintes
        if omega <= 0 or alpha < 0 or beta < 0 or gamma < 0:
            return 1e10
        if alpha + beta + gamma / 2 >= 1:
            return 1e10

        variance = self._compute_variance(params)
        variance = np.maximum(variance, 1e-10)

        log_likelihood = -0.5 * np.sum(
            np.log(variance) + self.returns ** 2 / variance
        )

        if not np.isfinite(log_likelihood):
            return 1e10

        return -log_likelihood

    def fit(self):
        """
        Estime les paramètres GJR-GARCH par maximum de vraisemblance.
        """
        var_emp = np.var(self.returns)

        alpha_init = 0.05
        beta_init = 0.85
        gamma_init = 0.05
        omega_init = var_emp * (1 - alpha_init - beta_init - gamma_init / 2)

        initial_params = [omega_init, alpha_init, beta_init, gamma_init]

        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=[
                (1e-10, None),   # omega > 0
                (0, 0.999),      # alpha >= 0
                (0, 0.999),      # beta >= 0
                (0, 0.999),      # gamma >= 0
            ],
            options={'maxiter': 1000}
        )

        self.omega = result.x[0]
        self.alpha = result.x[1]
        self.beta = result.x[2]
        self.gamma = result.x[3]

        variance = self._compute_variance(result.x)
        self.conditional_vol = np.sqrt(variance)
        self._log_likelihood = -result.fun

        persistence = self.alpha + self.beta + self.gamma / 2

        # Variance long-terme
        if (1 - persistence) > 0.001:
            long_term_var = self.omega / (1 - persistence)
        else:
            long_term_var = var_emp

        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'persistence': persistence,
            'long_term_vol': np.sqrt(long_term_var),
            'log_likelihood': self._log_likelihood,
            'converged': result.success,
        }

    def forecast(self, horizon=1):
        """
        Prévision de la volatilité future GJR-GARCH.

        Pour les prévisions multi-pas, E[I_{t+h}] = 0.5 (symétrie),
        donc le coefficient effectif est α + γ/2.
        """
        if self.omega is None:
            raise ValueError("Le modèle doit être estimé d'abord (appeler fit())")

        persistence = self.alpha + self.beta + self.gamma / 2

        if (1 - persistence) > 0.001:
            var_long_term = self.omega / (1 - persistence)
        else:
            var_long_term = self.conditional_vol[-1] ** 2

        var_current = self.conditional_vol[-1] ** 2

        forecasts = np.zeros(horizon)
        for h in range(horizon):
            forecasts[h] = var_long_term + (persistence ** (h + 1)) * (var_current - var_long_term)

        return np.sqrt(np.maximum(forecasts, 1e-10))

    def get_volatility_series(self):
        """Retourne la série des volatilités conditionnelles."""
        if self.conditional_vol is None:
            raise ValueError("Le modèle doit être estimé d'abord")
        return self.conditional_vol


def compare_garch_models(returns):
    """
    Compare GARCH(1,1), EGARCH et GJR-GARCH sur les mêmes données.

    Retourne un dict avec pour chaque modèle:
        - paramètres estimés
        - log-vraisemblance
        - AIC = -2*LL + 2*k
        - BIC = -2*LL + k*ln(n)
        - persistance

    Paramètres:
    -----------
    returns : array-like
        Rendements historiques

    Retourne:
    ---------
    dict : résultats par modèle {'GARCH': {...}, 'EGARCH': {...}, 'GJR-GARCH': {...}}
    """
    returns = np.array(returns)
    n = len(returns)
    results = {}

    # GARCH(1,1) - 3 paramètres
    garch = GARCHModel(returns)
    garch_fit = garch.fit()
    k_garch = 3
    results['GARCH'] = {
        'params': {'omega': garch_fit['omega'], 'alpha': garch_fit['alpha'],
                   'beta': garch_fit['beta']},
        'log_likelihood': garch_fit['log_likelihood'],
        'n_params': k_garch,
        'AIC': -2 * garch_fit['log_likelihood'] + 2 * k_garch,
        'BIC': -2 * garch_fit['log_likelihood'] + k_garch * np.log(n),
        'persistence': garch_fit['persistence'],
        'converged': garch_fit['converged'],
    }

    # EGARCH - 4 paramètres
    egarch = EGARCHModel(returns)
    egarch_fit = egarch.fit()
    k_egarch = 4
    results['EGARCH'] = {
        'params': {'omega': egarch_fit['omega'], 'alpha': egarch_fit['alpha'],
                   'beta': egarch_fit['beta'], 'gamma': egarch_fit['gamma']},
        'log_likelihood': egarch_fit['log_likelihood'],
        'n_params': k_egarch,
        'AIC': -2 * egarch_fit['log_likelihood'] + 2 * k_egarch,
        'BIC': -2 * egarch_fit['log_likelihood'] + k_egarch * np.log(n),
        'persistence': egarch_fit['persistence'],
        'converged': egarch_fit['converged'],
    }

    # GJR-GARCH - 4 paramètres
    gjr = GJRGARCHModel(returns)
    gjr_fit = gjr.fit()
    k_gjr = 4
    results['GJR-GARCH'] = {
        'params': {'omega': gjr_fit['omega'], 'alpha': gjr_fit['alpha'],
                   'beta': gjr_fit['beta'], 'gamma': gjr_fit['gamma']},
        'log_likelihood': gjr_fit['log_likelihood'],
        'n_params': k_gjr,
        'AIC': -2 * gjr_fit['log_likelihood'] + 2 * k_gjr,
        'BIC': -2 * gjr_fit['log_likelihood'] + k_gjr * np.log(n),
        'persistence': gjr_fit['persistence'],
        'converged': gjr_fit['converged'],
    }

    return results


def ewma_volatility(returns, lambda_param=0.94):
    """
    Calcule la volatilité EWMA (Exponentially Weighted Moving Average).

    Modèle EWMA (cas particulier de GARCH avec ω=0):
        σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}

    RiskMetrics utilise λ = 0.94 pour les données journalières.

    Paramètres:
    -----------
    returns : array-like
        Rendements
    lambda_param : float
        Paramètre de lissage (défaut: 0.94 = RiskMetrics)

    Retourne:
    ---------
    np.array : volatilités EWMA
    """
    returns = np.array(returns)
    n = len(returns)
    variance = np.zeros(n)

    # Initialisation
    variance[0] = returns[0]**2

    # Récurrence EWMA
    for t in range(1, n):
        variance[t] = lambda_param * variance[t-1] + (1 - lambda_param) * returns[t-1]**2

    return np.sqrt(variance)


# Tests si exécuté directement
if __name__ == "__main__":
    # Générer des données avec volatilité variable (modèle GARCH simulé)
    np.random.seed(42)
    n = 1000

    # Vrais paramètres
    true_omega = 0.00001
    true_alpha = 0.10
    true_beta = 0.85

    # Simulation GARCH
    returns = np.zeros(n)
    variance = np.zeros(n)
    variance[0] = true_omega / (1 - true_alpha - true_beta)

    for t in range(1, n):
        variance[t] = true_omega + true_alpha * returns[t-1]**2 + true_beta * variance[t-1]
        returns[t] = np.sqrt(variance[t]) * np.random.standard_normal()

    print("=== Test GARCH(1,1) ===")
    print(f"\nVrais paramètres:")
    print(f"  ω = {true_omega:.6f}")
    print(f"  α = {true_alpha:.2f}")
    print(f"  β = {true_beta:.2f}")

    # Estimer le modèle
    model = GARCHModel(returns)
    result = model.fit()

    print(f"\nParamètres estimés:")
    print(f"  ω = {result['omega']:.6f}")
    print(f"  α = {result['alpha']:.4f}")
    print(f"  β = {result['beta']:.4f}")
    print(f"  Persistance (α+β) = {result['persistence']:.4f}")
    print(f"  Vol long-terme = {result['long_term_vol']*100:.2f}%")
    print(f"  Convergence: {result['converged']}")

    # Prévision
    print(f"\nPrévision de volatilité:")
    forecasts = model.forecast(horizon=5)
    for i, vol in enumerate(forecasts):
        print(f"  Jour +{i+1}: {vol*100:.4f}%")

    # Comparaison avec EWMA
    print(f"\n--- Comparaison avec EWMA (λ=0.94) ---")
    ewma_vol = ewma_volatility(returns, 0.94)
    garch_vol = model.get_volatility_series()

    print(f"Vol finale GARCH: {garch_vol[-1]*100:.4f}%")
    print(f"Vol finale EWMA:  {ewma_vol[-1]*100:.4f}%")
