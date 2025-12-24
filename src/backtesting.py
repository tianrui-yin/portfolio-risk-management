# -*- coding: utf-8 -*-
"""
Backtesting du VaR
==================
Tests statistiques pour valider les modèles de VaR.

Référence: Hull, Chapitre 22

Un modèle VaR est correct si le nombre de dépassements
(exceptions) correspond au niveau de confiance choisi.

Tests implémentés:
    - Test de Kupiec (couverture inconditionnelle)
    - Test de Christoffersen (indépendance des exceptions)
"""

import numpy as np
from scipy.stats import chi2


def count_exceptions(returns, var_estimates):
    """
    Compte le nombre d'exceptions (dépassements du VaR).

    Une exception se produit quand la perte réelle dépasse le VaR prévu.

    Paramètres:
    -----------
    returns : np.array
        Rendements réalisés
    var_estimates : np.array
        Estimations du VaR (positif = perte)

    Retourne:
    ---------
    dict : nombre et positions des exceptions
    """
    # Une exception = perte réelle > VaR
    # perte réelle = -rendement (car rendement négatif = perte)
    losses = -np.array(returns)
    var = np.array(var_estimates)

    # Exception quand la perte dépasse le VaR
    exceptions = losses > var

    return {
        'n_exceptions': np.sum(exceptions),
        'exception_rate': np.mean(exceptions),
        'exception_indices': np.where(exceptions)[0],
        'total_observations': len(returns)
    }


def kupiec_test(returns, var_estimates, confidence=0.95):
    """
    Test de Kupiec (1995) - Proportion of Failures (POF).

    Ce test vérifie si le taux d'exceptions observé correspond
    au niveau de confiance du modèle VaR.

    Hypothèses:
        H0: Le taux d'exceptions = (1 - confidence)
        H1: Le taux d'exceptions ≠ (1 - confidence)

    Statistique de test (Log-likelihood ratio):
        LR = 2 * [log(L1) - log(L0)]

    où L1 utilise le taux observé et L0 utilise le taux théorique.

    Sous H0, LR ~ χ²(1)

    Paramètres:
    -----------
    returns : np.array
        Rendements réalisés
    var_estimates : np.array
        Estimations du VaR
    confidence : float
        Niveau de confiance du VaR

    Retourne:
    ---------
    dict : résultats du test

    Exemple:
    --------
    >>> result = kupiec_test(returns, var, 0.95)
    >>> if result['passed']:
    ...     print("VaR validé")
    """
    # Compter les exceptions
    exc = count_exceptions(returns, var_estimates)
    n = exc['total_observations']
    x = exc['n_exceptions']

    # Taux théorique (sous H0)
    p0 = 1 - confidence

    # Taux observé
    p_obs = x / n if n > 0 else 0

    # Log-likelihood ratio
    # LR = 2 * [x*log(p_obs/p0) + (n-x)*log((1-p_obs)/(1-p0))]

    # Éviter log(0)
    epsilon = 1e-10
    p_obs = max(epsilon, min(1 - epsilon, p_obs))

    if x == 0:
        # Pas d'exception: LR simplifié
        lr_statistic = -2 * n * np.log(1 - p0)
    elif x == n:
        # Toutes exceptions: LR simplifié
        lr_statistic = -2 * n * np.log(p0)
    else:
        # Formule complète
        lr_statistic = 2 * (
            x * np.log(p_obs / p0) +
            (n - x) * np.log((1 - p_obs) / (1 - p0))
        )

    # P-value (χ² avec 1 degré de liberté)
    p_value = 1 - chi2.cdf(lr_statistic, df=1)

    # Seuil de rejet à 5%
    critical_value = chi2.ppf(0.95, df=1)
    passed = lr_statistic < critical_value

    return {
        'test_name': 'Kupiec (POF)',
        'n_observations': n,
        'n_exceptions': x,
        'expected_exceptions': n * p0,
        'exception_rate_observed': p_obs,
        'exception_rate_expected': p0,
        'lr_statistic': lr_statistic,
        'critical_value': critical_value,
        'p_value': p_value,
        'passed': passed,
        'conclusion': 'VaR validé' if passed else 'VaR rejeté'
    }


def christoffersen_test(returns, var_estimates, confidence=0.95):
    """
    Test de Christoffersen (1998) - Test d'indépendance.

    Ce test vérifie que les exceptions ne sont pas groupées
    (c'est-à-dire qu'elles sont indépendantes dans le temps).

    Le test combine:
        1. Test de couverture (comme Kupiec)
        2. Test d'indépendance

    Si les exceptions sont groupées, cela suggère que le modèle
    ne réagit pas assez vite aux changements de volatilité.

    Statistique:
        LR_ind = 2 * [log(L1) - log(L0)]

    où L1 utilise les probabilités de transition observées
    et L0 suppose l'indépendance.

    Paramètres:
    -----------
    returns : np.array
        Rendements réalisés
    var_estimates : np.array
        Estimations du VaR
    confidence : float
        Niveau de confiance

    Retourne:
    ---------
    dict : résultats du test
    """
    # Identifier les exceptions
    exc = count_exceptions(returns, var_estimates)
    n = exc['total_observations']
    exceptions = np.zeros(n, dtype=int)
    exceptions[exc['exception_indices']] = 1

    # Matrice de transition
    # n_ij = nombre de transitions de l'état i à l'état j
    # État 0 = pas d'exception, État 1 = exception
    n_00 = 0  # Pas d'exception puis pas d'exception
    n_01 = 0  # Pas d'exception puis exception
    n_10 = 0  # Exception puis pas d'exception
    n_11 = 0  # Exception puis exception

    for t in range(1, n):
        if exceptions[t-1] == 0 and exceptions[t] == 0:
            n_00 += 1
        elif exceptions[t-1] == 0 and exceptions[t] == 1:
            n_01 += 1
        elif exceptions[t-1] == 1 and exceptions[t] == 0:
            n_10 += 1
        else:  # 1 -> 1
            n_11 += 1

    # Probabilités de transition observées
    # P(exception | pas d'exception précédente)
    if (n_00 + n_01) > 0:
        pi_01 = n_01 / (n_00 + n_01)
    else:
        pi_01 = 0

    # P(exception | exception précédente)
    if (n_10 + n_11) > 0:
        pi_11 = n_11 / (n_10 + n_11)
    else:
        pi_11 = 0

    # Probabilité globale d'exception
    if n > 1:
        pi = (n_01 + n_11) / (n - 1)
    else:
        pi = 0

    # Statistique LR pour l'indépendance
    epsilon = 1e-10

    # Log-likelihood sous H0 (indépendance): toutes les transitions ont même prob
    if pi > epsilon and pi < 1 - epsilon:
        log_l0 = (n_01 + n_11) * np.log(pi) + (n_00 + n_10) * np.log(1 - pi)
    else:
        log_l0 = 0

    # Log-likelihood sous H1 (dépendance)
    log_l1 = 0
    if pi_01 > epsilon and pi_01 < 1 - epsilon:
        log_l1 += n_01 * np.log(pi_01) + n_00 * np.log(1 - pi_01)
    if pi_11 > epsilon and pi_11 < 1 - epsilon:
        log_l1 += n_11 * np.log(pi_11) + n_10 * np.log(1 - pi_11)

    # Statistique LR indépendance
    lr_ind = 2 * (log_l1 - log_l0)
    lr_ind = max(0, lr_ind)  # Éviter les valeurs négatives dues aux erreurs numériques

    # Test de Kupiec (couverture)
    kupiec_result = kupiec_test(returns, var_estimates, confidence)
    lr_cov = kupiec_result['lr_statistic']

    # Test combiné (couverture + indépendance)
    lr_combined = lr_cov + lr_ind

    # P-values
    p_value_ind = 1 - chi2.cdf(lr_ind, df=1)
    p_value_combined = 1 - chi2.cdf(lr_combined, df=2)

    # Seuils critiques
    critical_ind = chi2.ppf(0.95, df=1)
    critical_combined = chi2.ppf(0.95, df=2)

    # Résultats
    passed_ind = lr_ind < critical_ind
    passed_combined = lr_combined < critical_combined

    return {
        'test_name': 'Christoffersen (Indépendance + Couverture)',
        'n_observations': n,
        'n_exceptions': exc['n_exceptions'],

        # Matrice de transition
        'transition_matrix': {
            'n_00': n_00, 'n_01': n_01,
            'n_10': n_10, 'n_11': n_11
        },
        'pi_01': pi_01,  # P(exc | no exc)
        'pi_11': pi_11,  # P(exc | exc)

        # Test d'indépendance
        'lr_independence': lr_ind,
        'p_value_independence': p_value_ind,
        'passed_independence': passed_ind,

        # Test de couverture (Kupiec)
        'lr_coverage': lr_cov,
        'p_value_coverage': kupiec_result['p_value'],
        'passed_coverage': kupiec_result['passed'],

        # Test combiné
        'lr_combined': lr_combined,
        'critical_value_combined': critical_combined,
        'p_value_combined': p_value_combined,
        'passed_combined': passed_combined,

        'conclusion': 'VaR validé' if passed_combined else 'VaR rejeté'
    }


def backtest_report(returns, var_estimates, confidence=0.95, var_method=''):
    """
    Génère un rapport complet de backtesting.

    Paramètres:
    -----------
    returns : np.array
        Rendements réalisés
    var_estimates : np.array
        Estimations du VaR
    confidence : float
        Niveau de confiance
    var_method : str
        Nom de la méthode VaR utilisée

    Retourne:
    ---------
    str : rapport formaté
    """
    kupiec = kupiec_test(returns, var_estimates, confidence)
    christoffersen = christoffersen_test(returns, var_estimates, confidence)

    lines = [
        f"\n{'='*60}",
        f"RAPPORT DE BACKTESTING - VaR {confidence*100:.0f}%",
        f"Méthode: {var_method}" if var_method else "",
        f"{'='*60}",
        f"\nSTATISTIQUES GÉNÉRALES:",
        f"  Observations: {kupiec['n_observations']}",
        f"  Exceptions observées: {kupiec['n_exceptions']}",
        f"  Exceptions attendues: {kupiec['expected_exceptions']:.1f}",
        f"  Taux observé: {kupiec['exception_rate_observed']*100:.2f}%",
        f"  Taux attendu: {kupiec['exception_rate_expected']*100:.2f}%",
        f"\nTEST DE KUPIEC (Couverture inconditionnelle):",
        f"  Statistique LR: {kupiec['lr_statistic']:.4f}",
        f"  Valeur critique (5%): {kupiec['critical_value']:.4f}",
        f"  P-value: {kupiec['p_value']:.4f}",
        f"  Résultat: {'✓ PASSÉ' if kupiec['passed'] else '✗ ÉCHOUÉ'}",
        f"\nTEST DE CHRISTOFFERSEN (Indépendance):",
        f"  Statistique LR (indépendance): {christoffersen['lr_independence']:.4f}",
        f"  P-value: {christoffersen['p_value_independence']:.4f}",
        f"  Résultat: {'✓ PASSÉ' if christoffersen['passed_independence'] else '✗ ÉCHOUÉ'}",
        f"\nTEST COMBINÉ:",
        f"  Statistique LR: {christoffersen['lr_combined']:.4f}",
        f"  Valeur critique (5%): {christoffersen['critical_value_combined']:.4f}",
        f"  P-value: {christoffersen['p_value_combined']:.4f}",
        f"  Résultat: {'✓ PASSÉ' if christoffersen['passed_combined'] else '✗ ÉCHOUÉ'}",
        f"\n{'='*60}",
        f"CONCLUSION: {christoffersen['conclusion'].upper()}",
        f"{'='*60}"
    ]

    return '\n'.join(lines)


def exception_clustering(returns, var_estimates, confidence=0.95):
    """
    Analyse du clustering des exceptions VaR.

    Étudie si les exceptions (dépassements du VaR) se regroupent
    dans le temps, ce qui indiquerait que le modèle ne réagit pas
    assez vite aux changements de volatilité.

    Paramètres:
    -----------
    returns : np.array
        Rendements réalisés
    var_estimates : np.array
        Estimations du VaR (positif = perte)
    confidence : float
        Niveau de confiance

    Retourne:
    ---------
    dict : analyse du clustering
        - exception_dates: indices des exceptions
        - max_consecutive: plus longue série consécutive
        - cluster_count: nombre de clusters
        - average_cluster_size: taille moyenne des clusters
        - independence_test: LR et p-value d'indépendance
        - interpretation: interprétation textuelle
    """
    exc = count_exceptions(returns, var_estimates)
    exception_indices = exc['exception_indices']
    n = exc['total_observations']

    if len(exception_indices) == 0:
        return {
            'exception_dates': np.array([], dtype=int),
            'max_consecutive': 0,
            'cluster_count': 0,
            'average_cluster_size': 0.0,
            'independence_test': {'lr_statistic': 0.0, 'p_value': 1.0},
            'interpretation': 'Aucune exception détectée.',
        }

    # Construire la série binaire d'exceptions
    exceptions_binary = np.zeros(n, dtype=int)
    exceptions_binary[exception_indices] = 1

    # Trouver les clusters (groupes consécutifs d'exceptions)
    clusters = []
    current_cluster = 0
    in_cluster = False

    for t in range(n):
        if exceptions_binary[t] == 1:
            if not in_cluster:
                in_cluster = True
                current_cluster = 1
            else:
                current_cluster += 1
        else:
            if in_cluster:
                clusters.append(current_cluster)
                in_cluster = False
                current_cluster = 0

    # Fermer le dernier cluster si nécessaire
    if in_cluster:
        clusters.append(current_cluster)

    max_consecutive = max(clusters) if clusters else 0
    cluster_count = len(clusters)
    average_cluster_size = np.mean(clusters) if clusters else 0.0

    # Test d'indépendance de Christoffersen (partie indépendance seule)
    chris = christoffersen_test(returns, var_estimates, confidence)
    independence_test = {
        'lr_statistic': chris['lr_independence'],
        'p_value': chris['p_value_independence'],
    }

    # Interprétation
    if max_consecutive >= 3 or chris['p_value_independence'] < 0.05:
        interpretation = (
            f"Clustering détecté: {cluster_count} cluster(s), "
            f"max consécutif = {max_consecutive}. "
            f"Le modèle ne s'adapte pas assez vite aux changements de volatilité."
        )
    else:
        interpretation = (
            f"Pas de clustering significatif: {cluster_count} cluster(s), "
            f"max consécutif = {max_consecutive}. "
            f"Les exceptions semblent indépendantes."
        )

    return {
        'exception_dates': exception_indices,
        'max_consecutive': max_consecutive,
        'cluster_count': cluster_count,
        'average_cluster_size': average_cluster_size,
        'independence_test': independence_test,
        'interpretation': interpretation,
    }


def lookback_sensitivity(returns, confidence=0.95,
                          lookback_periods=None):
    """
    Analyse de sensibilité au lookback period.

    Teste comment le VaR et les résultats de backtesting changent
    en fonction de la fenêtre d'estimation.

    Paramètres:
    -----------
    returns : np.array
        Rendements historiques
    confidence : float
        Niveau de confiance
    lookback_periods : list
        Périodes de lookback à tester (défaut: [125, 250, 500, 750, 1000])

    Retourne:
    ---------
    pd.DataFrame : lookback, var_95, var_99, exception_rate, kupiec_pvalue
    """
    import pandas as pd
    from scipy.stats import norm

    if lookback_periods is None:
        lookback_periods = [125, 250, 500, 750, 1000]

    returns = np.array(returns)
    n = len(returns)
    z_95 = norm.ppf(0.95)
    z_99 = norm.ppf(0.99)

    rows = []
    for lookback in lookback_periods:
        if lookback >= n:
            continue

        # Utiliser les 'lookback' derniers jours pour estimer σ
        recent = returns[-lookback:]
        sigma = np.std(recent, ddof=1)

        var_95 = z_95 * sigma
        var_99 = z_99 * sigma

        # Backtesting: VaR glissant sur la période hors-échantillon
        n_out = n - lookback
        if n_out < 10:
            rows.append({
                'lookback': lookback,
                'var_95': var_95,
                'var_99': var_99,
                'exception_rate': np.nan,
                'kupiec_pvalue': np.nan,
            })
            continue

        # VaR glissant
        var_rolling = np.zeros(n_out)
        z_conf = norm.ppf(confidence)
        for t in range(n_out):
            sigma_t = np.std(returns[t:t + lookback], ddof=1)
            var_rolling[t] = z_conf * sigma_t

        returns_test = returns[lookback:]
        losses = -returns_test
        exceptions = losses > var_rolling
        exception_rate = np.mean(exceptions)

        # Kupiec test
        kupiec = kupiec_test(returns_test, var_rolling, confidence)

        rows.append({
            'lookback': lookback,
            'var_95': var_95,
            'var_99': var_99,
            'exception_rate': exception_rate,
            'kupiec_pvalue': kupiec['p_value'],
        })

    return pd.DataFrame(rows)


def confidence_sensitivity(returns, var_estimates_by_confidence,
                            confidence_levels=None):
    """
    Analyse de sensibilité au niveau de confiance.

    Paramètres:
    -----------
    returns : np.array
        Rendements réalisés
    var_estimates_by_confidence : dict
        {confidence_level: var_estimates_array}
    confidence_levels : list
        Niveaux de confiance à tester

    Retourne:
    ---------
    dict : résultats par niveau de confiance
        {0.95: {'exception_rate': ..., 'kupiec': ..., 'expected_rate': ...}, ...}
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.975, 0.99]

    returns = np.array(returns)
    results = {}

    for conf in confidence_levels:
        if conf not in var_estimates_by_confidence:
            continue

        var_estimates = np.array(var_estimates_by_confidence[conf])
        exc = count_exceptions(returns, var_estimates)

        kupiec = kupiec_test(returns, var_estimates, conf)

        results[conf] = {
            'exception_rate': exc['exception_rate'],
            'n_exceptions': exc['n_exceptions'],
            'expected_rate': 1 - conf,
            'expected_exceptions': len(returns) * (1 - conf),
            'kupiec': kupiec,
        }

    return results


# Tests si exécuté directement
if __name__ == "__main__":
    # Générer des données de test
    np.random.seed(42)
    n_days = 500

    # Simuler des rendements normaux
    returns = np.random.normal(0, 0.02, n_days)

    # Ajouter quelques événements extrêmes
    returns[50] = -0.08
    returns[200] = -0.07
    returns[350] = -0.09

    # Estimer le VaR (méthode simple: rolling std)
    var_estimates = np.zeros(n_days)
    window = 60

    for t in range(window, n_days):
        std = np.std(returns[t-window:t])
        var_estimates[t] = 1.645 * std  # VaR 95%

    # Backtesting sur la partie avec estimations
    returns_test = returns[window:]
    var_test = var_estimates[window:]

    print("=== Test Backtesting VaR ===")

    # Test de Kupiec
    kupiec = kupiec_test(returns_test, var_test, 0.95)
    print(f"\nTest de Kupiec:")
    print(f"  Exceptions: {kupiec['n_exceptions']} / {kupiec['n_observations']}")
    print(f"  Attendues: {kupiec['expected_exceptions']:.1f}")
    print(f"  Statistique LR: {kupiec['lr_statistic']:.4f}")
    print(f"  P-value: {kupiec['p_value']:.4f}")
    print(f"  Résultat: {kupiec['conclusion']}")

    # Test de Christoffersen
    chris = christoffersen_test(returns_test, var_test, 0.95)
    print(f"\nTest de Christoffersen:")
    print(f"  LR indépendance: {chris['lr_independence']:.4f}")
    print(f"  LR combiné: {chris['lr_combined']:.4f}")
    print(f"  Résultat: {chris['conclusion']}")

    # Rapport complet
    print(backtest_report(returns_test, var_test, 0.95, 'Rolling Window 60j'))
