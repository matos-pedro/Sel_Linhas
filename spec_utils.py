from hapi import fetch, getColumn, partitionSum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def fator_correcao_termo(nu, T, T_ref):
    c2 = 1.4387769
    return (1 - np.exp(-c2 * nu / T)) / (1 - np.exp(-c2 * nu / T_ref))


def estimar_T(I1, I2, S1, S2, E1, E2, nu1, nu2, T_ref=296, T_chute=1000, tol=1, max_iter=10):
    c2 = 1.4387769
    if I1 <= 0 or I2 <= 0 or E1 == E2:
        return np.nan

    T = T_chute
    for _ in range(max_iter):
        F1 = fator_correcao_termo(nu1, T, T_ref)
        F2 = fator_correcao_termo(nu2, T, T_ref)

        R = (I1 / I2) * (S2 / S1) * (F2 / F1) * np.exp(-c2 * (E1 - E2) / T_ref)
        if R <= 0:
            return np.nan

        T_new = -c2 * (E1 - E2) / np.log(R)
        if np.abs(T_new - T) < tol:
            return T_new
        T = T_new

    return T


def analisar_sigma_T(T, range_1, range_2, mol_id=1, iso_id=1, T_ref=296, Nl=10, nm=True, erro_rel_I=0.05, K=10):
    c2 = 1.4387769

    def calc_intensity(S_ref, E_lower, nu, T, Q_T, Q_T_ref, T_ref):
        fator_boltzmann = np.exp(-c2 * E_lower * (1/T - 1/T_ref))
        fator_termico   = (1 - np.exp(-c2 * nu / T)) / (1 - np.exp(-c2 * nu / T_ref))
        return S_ref * (Q_T_ref / Q_T) * fator_boltzmann * fator_termico

    if nm:
        range_1 = (1e7 / range_1[1], 1e7 / range_1[0])
        range_2 = (1e7 / range_2[1], 1e7 / range_2[0])
    else:
        range_1 = (range_1[0], range_1[1])
        range_2 = (range_2[0], range_2[1])

    nu_min = min(range_1[0], range_2[0])
    nu_max = max(range_1[1], range_2[1])
    table_name = 'H2O_temp_analysis'

    fetch(table_name, mol_id, iso_id, nu_min, nu_max)

    nu      = np.array(getColumn(table_name, 'nu'))
    S_ref   = np.array(getColumn(table_name, 'sw'))
    E_lower = np.array(getColumn(table_name, 'elower'))

    Q_T = partitionSum(mol_id, iso_id, T)
    Q_T_ref = partitionSum(mol_id, iso_id, T_ref)

    I_T = calc_intensity(S_ref, E_lower, nu, T, Q_T, Q_T_ref, T_ref)
    I_max = np.max(I_T)
    erro_abs = erro_rel_I * I_max

    idx_1 = np.where((nu >= range_1[0]) & (nu <= range_1[1]))[0]
    idx_2 = np.where((nu >= range_2[0]) & (nu <= range_2[1]))[0]

    top_1_idx = idx_1[np.argsort(I_T[idx_1])[-Nl:]]
    top_2_idx = idx_2[np.argsort(I_T[idx_2])[-Nl:]]

    top_1_idx = top_1_idx[np.argsort(nu[top_1_idx])]
    top_2_idx = top_2_idx[np.argsort(nu[top_2_idx])]

    sigma_T_matrix = np.zeros((Nl, Nl))

    for i, idx_i in enumerate(top_1_idx):
        for j, idx_j in enumerate(top_2_idx):
            I1 = I_T[idx_i]
            I2 = I_T[idx_j]

            S1 = S_ref[idx_i]
            S2 = S_ref[idx_j]
            nu1 = nu[idx_i]
            nu2 = nu[idx_j]
            E1 = E_lower[idx_i]
            E2 = E_lower[idx_j]

            T_estimates = []
            for _ in range(K):
                I1_pert = I1 + np.random.normal(0, erro_abs)
                I2_pert = I2 + np.random.normal(0, erro_abs)
                T_hat = estimar_T(I1_pert, I2_pert, S1, S2, E1, E2, nu1, nu2, T_ref=T_ref, T_chute=T)
                T_estimates.append(T_hat)

            T_estimates = np.array(T_estimates)
            T_validos = T_estimates[~np.isnan(T_estimates)]
            sigma_T = np.sqrt(np.mean((T_validos - T)**2)) if len(T_validos) > 1 else np.nan

            sigma_T_matrix[i, j] = sigma_T

    # Plot do mapa de calor com escala log
    fig, ax = plt.subplots(figsize=(10, 8))
    sigma_T_matrix_plot = np.where(sigma_T_matrix > 0, sigma_T_matrix, np.nan)
    im = ax.imshow(sigma_T_matrix_plot, cmap='coolwarm', aspect='auto', norm=LogNorm(vmin=np.nanmin(sigma_T_matrix_plot), vmax=np.nanmax(sigma_T_matrix_plot)))

    ax.set_xticks(np.arange(Nl))
    ax.set_yticks(np.arange(Nl))

    if nm:
        xticks = np.round(1e7 / nu[top_2_idx], 3)
        yticks = np.round(1e7 / nu[top_1_idx], 3)
        ax.set_xlabel('Linhas da Região 2 (nm)')
        ax.set_ylabel('Linhas da Região 1 (nm)')
    else:
        xticks = np.round(nu[top_2_idx], 3)
        yticks = np.round(nu[top_1_idx], 3)
        ax.set_xlabel('Linhas da Região 2 (cm⁻¹)')
        ax.set_ylabel('Linhas da Região 1 (cm⁻¹)')

    ax.set_xticklabels([f"{v:.3f}" for v in xticks], rotation=45, ha='right')
    ax.set_yticklabels([f"{v:.3f}" for v in yticks])
    ax.set_title(f'Desvio σ(T) a T = {T} K, (min,max) = ({np.nanmin(sigma_T_matrix):.0f}, {np.nanmax(sigma_T_matrix):.2e}) K \nRuído Relativo Adicionado : {erro_rel_I*100}% da maior intensidade')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('σ(T) (K)')
    plt.tight_layout()
    plt.grid(alpha=0.0)

    return sigma_T_matrix, nu[top_1_idx], nu[top_2_idx]
