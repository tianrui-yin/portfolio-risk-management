# -*- coding: utf-8 -*-
"""
VaR Non-Linéaire : Delta-Gamma et Full Revaluation
====================================================
Méthodes de VaR pour portefeuilles contenant des options (risque non-linéaire).

Trois approches implémentées :
    1. Delta-Normal VaR : approximation linéaire (utilise uniquement Delta)
    2. Delta-Gamma VaR  : approximation quadratique (Cornish-Fisher ou moment-matching)
    3. Full Revaluation MC VaR : repricing complet par Black-Scholes sous chaque scénario

Référence : Hull, Options, Futures and Other Derivatives, Chapitre 22.
"""

import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp


class NonLinearVaR:
    """
    VaR pour portefeuilles contenant des options (risque non-linéaire).

    Le P&L d'un portefeuille avec options est non-linéaire dans les facteurs
    de risque sous-jacents. Le VaR delta-normal sous-estime le risque car il
    ignore la convexité (gamma). Le delta-gamma VaR et le full revaluation
    capturent cette non-linéarité.

    Attributs:
    ----------
    positions : list[dict]
        Liste des positions, chacune contenant :
        - 'type' : 'stock' ou 'option'
        - 'quantity' : nombre d'unités
        - Pour stocks : 'price', 'sigma' (vol annuelle)
        - Pour options : 'S', 'K', 'T', 'r', 'sigma', 'option_type' ('call'/'put')
    """

    def __init__(self, positions):
        """
        Initialise le calculateur de VaR non-linéaire.

        Paramètres:
        -----------
        positions : list[dict]
            Liste des positions du portefeuille.
        """
        self.positions = positions
        self._validate_positions()

    def _validate_positions(self):
        """Valide la structure des positions."""
        for i, pos in enumerate(self.positions):
            if pos['type'] not in ('stock', 'option'):
                raise ValueError(
                    f"Position {i} : type inconnu '{pos['type']}'. "
                    f"Attendu 'stock' ou 'option'."
                )
            if pos['type'] == 'stock':
                for key in ('quantity', 'price', 'sigma'):
                    if key not in pos:
                        raise ValueError(
                            f"Position {i} (stock) : clé manquante '{key}'."
                        )
            elif pos['type'] == 'option':
                for key in ('quantity', 'S', 'K', 'T', 'r', 'sigma', 'option_type'):
                    if key not in pos:
                        raise ValueError(
                            f"Position {i} (option) : clé manquante '{key}'."
                        )

    # ================================================================
    # Black-Scholes helpers
    # ================================================================

    @staticmethod
    def bs_price(S, K, T, r, sigma, option_type='call'):
        """
        Prix Black-Scholes d'une option européenne.

        Paramètres:
        -----------
        S : float     — prix du sous-jacent
        K : float     — prix d'exercice
        T : float     — maturité (en années)
        r : float     — taux sans risque
        sigma : float — volatilité annuelle
        option_type : str — 'call' ou 'put'

        Retourne:
        ---------
        float : prix de l'option
        """
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if option_type == 'call':
            return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:
            return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def bs_delta(S, K, T, r, sigma, option_type='call'):
        """
        Delta Black-Scholes d'une option européenne.

        Delta = ∂C/∂S (call) ou ∂P/∂S (put).

        Retourne:
        ---------
        float : delta de l'option
        """
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0

    @staticmethod
    def bs_gamma(S, K, T, r, sigma, option_type='call'):
        """
        Gamma Black-Scholes d'une option européenne.

        Gamma = ∂²C/∂S² = ∂²P/∂S². Identique pour call et put.

        Retourne:
        ---------
        float : gamma de l'option (toujours positif)
        """
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        return norm.pdf(d1) / (S * sigma * sqrt(T))

    # ================================================================
    # Portfolio Greeks aggregation
    # ================================================================

    def _portfolio_greeks(self, horizon=1):
        """
        Calcule le delta et gamma agrégés du portefeuille, ainsi que
        le paramètre de dispersion σ_S = S * σ * sqrt(horizon/252).

        Pour un portefeuille à sous-jacent unique (hypothèse simplificatrice),
        on agrège :
            Δ_portfolio = Σ_i quantity_i * Δ_i
            Γ_portfolio = Σ_i quantity_i * Γ_i

        Retourne:
        ---------
        dict avec 'delta', 'gamma', 'S', 'sigma_S' (dispersion en valeur absolue)
        """
        total_delta = 0.0
        total_gamma = 0.0
        # Utiliser le S et sigma du premier actif comme référence
        ref_S = None
        ref_sigma = None

        for pos in self.positions:
            q = pos['quantity']
            if pos['type'] == 'stock':
                # Stock : delta = 1 par action, gamma = 0
                total_delta += q * 1.0
                if ref_S is None:
                    ref_S = pos['price']
                    ref_sigma = pos['sigma']
            elif pos['type'] == 'option':
                S = pos['S']
                K = pos['K']
                T = pos['T']
                r = pos['r']
                sigma = pos['sigma']
                opt_type = pos['option_type']

                delta = self.bs_delta(S, K, T, r, sigma, opt_type)
                gamma = self.bs_gamma(S, K, T, r, sigma, opt_type)

                total_delta += q * delta
                total_gamma += q * gamma

                if ref_S is None:
                    ref_S = S
                    ref_sigma = sigma

        # Dispersion du sous-jacent sur l'horizon (en valeur absolue)
        dt = horizon / 252.0
        sigma_S = ref_S * ref_sigma * sqrt(dt)

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'S': ref_S,
            'sigma': ref_sigma,
            'sigma_S': sigma_S,
        }

    # ================================================================
    # Delta-Normal VaR
    # ================================================================

    def delta_normal_var(self, confidence=0.95, horizon=1):
        """
        VaR delta-normal (approximation linéaire).

        VaR_dn = |Δ_portfolio| * σ_S * z_α

        Ne capture que le risque linéaire (delta). Sous-estime le VaR
        pour les portefeuilles avec une exposition gamma significative.

        Paramètres:
        -----------
        confidence : float — niveau de confiance (ex: 0.95)
        horizon : int      — horizon en jours

        Retourne:
        ---------
        float : VaR en valeur absolue (toujours positif)
        """
        greeks = self._portfolio_greeks(horizon)
        z_alpha = norm.ppf(confidence)
        var = abs(greeks['delta']) * greeks['sigma_S'] * z_alpha
        return var

    # ================================================================
    # Delta-Gamma VaR
    # ================================================================

    def delta_gamma_var(self, confidence=0.95, horizon=1, method='cornish-fisher'):
        """
        VaR delta-gamma (approximation quadratique).

        L'approximation du P&L :
            ΔP ≈ Δ·ΔS + ½Γ·(ΔS)²

        Deux méthodes :
        - 'cornish-fisher' : expansion de Cornish-Fisher pour distribution non-normale
        - 'moment-matching' : approximation de Johnson SU

        Paramètres:
        -----------
        confidence : float — niveau de confiance
        horizon : int      — horizon en jours
        method : str       — 'cornish-fisher' ou 'moment-matching'

        Retourne:
        ---------
        float : VaR en valeur absolue (toujours positif)
        """
        greeks = self._portfolio_greeks(horizon)
        delta = greeks['delta']
        gamma = greeks['gamma']
        sigma_S = greeks['sigma_S']

        if method == 'moment-matching':
            return self._delta_gamma_moment_matching(
                delta, gamma, sigma_S, confidence
            )
        elif method == 'cornish-fisher':
            return self._delta_gamma_cornish_fisher(
                delta, gamma, sigma_S, confidence
            )
        else:
            raise ValueError(f"Méthode inconnue : '{method}'. "
                             f"Attendu 'cornish-fisher' ou 'moment-matching'.")

    def _delta_gamma_moment_matching(self, delta, gamma, sigma_S, confidence):
        """
        Delta-Gamma VaR par moment-matching (Johnson SU).

        VaR = -(½Γ·σ_S² + Δ·σ_S·z_α + ½Γ·(σ_S·z_α)²)

        Le signe négatif car le VaR est une perte (positif).
        """
        z_alpha = norm.ppf(confidence)

        pnl = (
            0.5 * gamma * sigma_S ** 2
            + delta * sigma_S * (-z_alpha)
            + 0.5 * gamma * (sigma_S * z_alpha) ** 2
        )
        # VaR = perte maximale = -P&L au quantile bas
        var = -pnl
        return max(var, 0.0)

    def _delta_gamma_cornish_fisher(self, delta, gamma, sigma_S, confidence):
        """
        Delta-Gamma VaR par expansion de Cornish-Fisher.

        Moments de ΔP = Δ·ΔS + ½Γ·(ΔS)² avec ΔS ~ N(0, σ_S²) :
            E[ΔP]   = ½Γ·σ_S²
            Var[ΔP]  = Δ²·σ_S² + ½Γ²·σ_S⁴
            Skew[ΔP] = (6ΔΓ²σ_S⁶ + 2Γ³σ_S⁶) / Var[ΔP]^(3/2)

        Cornish-Fisher :
            z_cf = z_α + (z_α² - 1)·skew/6
            VaR  = -(E[ΔP] - z_cf·√Var[ΔP])
        """
        z_alpha = norm.ppf(confidence)

        # Moments de la distribution du P&L
        mean_pnl = 0.5 * gamma * sigma_S ** 2
        var_pnl = (delta ** 2) * (sigma_S ** 2) + 0.5 * (gamma ** 2) * (sigma_S ** 4)

        if var_pnl < 1e-20:
            # Pas de risque
            return 0.0

        std_pnl = sqrt(var_pnl)

        # Troisième moment centré : E[(ΔP - E[ΔP])³]
        # Pour ΔP = Δ·X + ½Γ·X² avec X ~ N(0, σ²) :
        # μ₃ = 6·Δ·Γ²·σ⁶ + 2·Γ³·σ⁶ (seulement les termes impairs)
        # En fait : μ₃ = E[(ΔP - E[ΔP])³]
        # ΔP - E[ΔP] = Δ·X + ½Γ·(X² - σ²)
        # μ₃ = E[(Δ·X + ½Γ·(X² - σ²))³]
        # En développant et utilisant E[X³]=0, E[X⁴]=3σ⁴, E[X⁵]=0, E[X⁶]=15σ⁶ :
        # μ₃ = 3·Δ²·Γ·σ⁴·(du terme croisé) + ... simplifié :
        # On calcule le skewness numériquement ou par formule exacte.
        # Formule exacte du 3ème moment centré :
        # μ₃ = 3Δ²Γσ⁴ + ½Γ³·(E[X⁶] - 3·E[X⁴]·σ² + 3σ⁴·σ² - σ⁶)
        # Simplifié : μ₃ = 6Δ²Γσ_S⁴ + 2Γ³σ_S⁶  (erreur commune, recalculons)
        #
        # Soit Y = Δ·X + ½Γ·(X² - σ²), X ~ N(0, σ²)
        # E[Y³] = Δ³·E[X³] + 3·Δ²·½Γ·E[X³·(X²-σ²)] + 3·Δ·(½Γ)²·E[X·(X²-σ²)²]
        #        + (½Γ)³·E[(X²-σ²)³]
        # E[X³] = 0, donc premier terme = 0
        # E[X³·(X²-σ²)] = E[X⁵] - σ²·E[X³] = 0
        # E[X·(X²-σ²)²] = E[X⁵ - 2σ²X³ + σ⁴X] = 0
        # E[(X²-σ²)³] = E[X⁶ - 3σ²X⁴ + 3σ⁴X² - σ⁶] = 15σ⁶ - 9σ⁶ + 3σ⁶ - σ⁶ = 8σ⁶
        # Donc μ₃ = (½Γ)³ · 8σ⁶ = Γ³σ⁶
        mu3 = (gamma ** 3) * (sigma_S ** 6)
        skew = mu3 / (std_pnl ** 3)

        # Expansion de Cornish-Fisher (termes d'ordre 1 en skewness)
        z_cf = z_alpha + (z_alpha ** 2 - 1) * skew / 6.0

        # VaR = -(E[ΔP] - z_cf · σ_ΔP)
        var = -(mean_pnl - z_cf * std_pnl)
        return max(var, 0.0)

    # ================================================================
    # Full Revaluation Monte Carlo VaR
    # ================================================================

    def full_revaluation_var(self, confidence=0.95, horizon=1,
                              n_simulations=50000, seed=42):
        """
        VaR par réévaluation complète Monte Carlo.

        Pour chaque scénario :
            1. Générer ΔS ~ N(0, (S·σ·√dt)²)
            2. Recalculer le prix de chaque position avec S_new = S + ΔS
            3. Calculer le P&L du portefeuille

        VaR = -percentile(P&L, 1 - confidence)

        Paramètres:
        -----------
        confidence : float   — niveau de confiance
        horizon : int        — horizon en jours
        n_simulations : int  — nombre de simulations
        seed : int           — graine aléatoire pour reproductibilité

        Retourne:
        ---------
        float : VaR en valeur absolue (toujours positif)
        """
        rng = np.random.RandomState(seed)

        greeks = self._portfolio_greeks(horizon)
        S = greeks['S']
        sigma_S = greeks['sigma_S']

        # Générer les variations du sous-jacent
        dS = rng.normal(0, sigma_S, n_simulations)
        S_new = S + dS

        # Calculer le P&L pour chaque scénario
        pnl = np.zeros(n_simulations)

        for pos in self.positions:
            q = pos['quantity']

            if pos['type'] == 'stock':
                # P&L linéaire : q * ΔS
                pnl += q * dS

            elif pos['type'] == 'option':
                K = pos['K']
                T = pos['T']
                r = pos['r']
                sigma = pos['sigma']
                opt_type = pos['option_type']

                # Prix actuel
                price_now = self.bs_price(
                    pos['S'], K, T, r, sigma, opt_type
                )

                # Repricing pour chaque scénario (vectorisé)
                prices_new = np.array([
                    self.bs_price(s, K, T, r, sigma, opt_type)
                    for s in S_new
                ])

                pnl += q * (prices_new - price_now)

        # VaR = quantile négatif
        alpha = 1 - confidence
        var = -np.percentile(pnl, alpha * 100)
        return max(var, 0.0)

    # ================================================================
    # Comparaison des méthodes
    # ================================================================

    def compare_methods(self, confidence=0.95, horizon=1):
        """
        Compare les trois méthodes de VaR non-linéaire.

        Retourne:
        ---------
        dict avec :
            - var_delta_normal, var_delta_gamma, var_full_reval
            - diff_dg_vs_dn  : écart relatif (DG - DN) / DN
            - diff_fr_vs_dn  : écart relatif (FR - DN) / DN
            - diff_fr_vs_dg  : écart relatif (FR - DG) / DG
        """
        var_dn = self.delta_normal_var(confidence, horizon)
        var_dg = self.delta_gamma_var(confidence, horizon)
        var_fr = self.full_revaluation_var(confidence, horizon)

        result = {
            'var_delta_normal': var_dn,
            'var_delta_gamma': var_dg,
            'var_full_reval': var_fr,
        }

        # Écarts relatifs (éviter division par zéro)
        if var_dn > 0:
            result['diff_dg_vs_dn'] = (var_dg - var_dn) / var_dn
            result['diff_fr_vs_dn'] = (var_fr - var_dn) / var_dn
        else:
            result['diff_dg_vs_dn'] = 0.0
            result['diff_fr_vs_dn'] = 0.0

        if var_dg > 0:
            result['diff_fr_vs_dg'] = (var_fr - var_dg) / var_dg
        else:
            result['diff_fr_vs_dg'] = 0.0

        return result


def example_portfolio():
    """
    Portefeuille d'exemple : 100 actions + 50 puts protecteurs.

    Stock : S=100, σ=25%
    Puts  : K=95, T=0.25, r=3%, σ=25%

    Ce portefeuille a une exposition gamma significative due aux puts,
    ce qui rend le VaR delta-normal insuffisant.
    """
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


# Tests si exécuté directement
if __name__ == "__main__":
    print("=" * 60)
    print("VaR Non-Linéaire : Delta-Gamma et Full Revaluation")
    print("=" * 60)

    nlv = example_portfolio()

    print("\nPortefeuille : 100 actions (S=100) + 50 puts (K=95, T=0.25)")
    print("-" * 60)

    greeks = nlv._portfolio_greeks()
    print(f"Delta portefeuille : {greeks['delta']:.4f}")
    print(f"Gamma portefeuille : {greeks['gamma']:.4f}")
    print(f"σ_S (1 jour)       : {greeks['sigma_S']:.4f}")

    for conf in [0.95, 0.99]:
        print(f"\n--- VaR {conf*100:.0f}% (1 jour) ---")
        result = nlv.compare_methods(confidence=conf)
        print(f"Delta-Normal     : {result['var_delta_normal']:>10.2f} €")
        print(f"Delta-Gamma (CF) : {result['var_delta_gamma']:>10.2f} €")
        print(f"Full Revaluation : {result['var_full_reval']:>10.2f} €")
        print(f"Écart DG vs DN   : {result['diff_dg_vs_dn']:>+9.1%}")
        print(f"Écart FR vs DN   : {result['diff_fr_vs_dn']:>+9.1%}")
