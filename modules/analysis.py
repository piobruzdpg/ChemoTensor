# modules/analysis.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
import tensorly as tl
from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import Constraint, ConstraintNorm
from scipy.linalg import pinv
import umap


# --- CUSTOM MCR CONSTRAINTS ---

class ConstraintClosure(Constraint):
    def __init__(self, axis=-1, copy=False):
        super().__init__(copy)
        self.axis = axis

    def transform(self, A):
        if self.copy: A = A.copy()
        sums = A.sum(axis=self.axis, keepdims=True)
        sums[sums == 0] = 1e-12  # Zabezpieczenie przed dzieleniem przez zero
        A /= sums
        return A


class ConstraintUnimodal(Constraint):
    def __init__(self, copy=False):
        super().__init__(copy)

    def transform(self, A):
        if self.copy: A = A.copy()
        for j in range(A.shape[1]):
            c = A[:, j]
            max_idx = np.argmax(c)
            # Wymuszenie spadku w lewo od maksimum
            for i in range(max_idx - 1, -1, -1):
                if c[i] > c[i + 1]: c[i] = c[i + 1]
            # Wymuszenie spadku w prawo od maksimum
            for i in range(max_idx + 1, len(c)):
                if c[i] > c[i - 1]: c[i] = c[i - 1]
            A[:, j] = c
        return A


# --- 1. BASIC METHODS (PCA, PLS, MANIFOLD) ---

def run_pca(X, n_components):
    """Runs Uncentered PCA using TruncatedSVD to strictly avoid internal mean-centering."""
    n_components = min(n_components, X.shape[0], X.shape[1] - 1)

    # Używamy TruncatedSVD, które w przeciwieństwie do PCA nie centruje danych
    svd = TruncatedSVD(n_components=n_components)
    scores = svd.fit_transform(X)

    return {
        'scores': scores,
        'loadings': svd.components_.T,
        'variance': svd.explained_variance_ratio_,
        'pca_model': svd,
        'scaler': None,
        'X_original': X
    }


def run_pca_reconstruction(pca_results, k):
    """Reconstructs data from k components."""
    svd = pca_results['pca_model']
    X_original = pca_results['X_original']
    scores = pca_results['scores']

    # Wyzerowanie komponentów powyżej k
    scores_k = np.copy(scores)
    scores_k[:, k:] = 0

    # Rekonstrukcja bezpośrednio przez odwrócenie SVD
    X_reconstructed = svd.inverse_transform(scores_k)

    residuals = X_original - X_reconstructed

    return {
        'X_reconstructed': X_reconstructed,
        'residuals': residuals,
        'k': k
    }

def run_pls(tensor, target_y, n_components=1):
    w, n, m = tensor.shape
    X = tensor.transpose(1, 2, 0).reshape((n * m), w)

    # Ignoruj puste próbki dla PLS
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(target_y)
    X_valid = X[valid_mask]
    y_valid = target_y[valid_mask]

    if X_valid.shape[0] < 2:
        raise ValueError("Not enough valid data points for PLS.")

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_valid, y_valid)
    return {'coefs': pls.coef_.flatten()}


def run_umap(X, n_components=2):
    model = umap.UMAP(n_components=n_components, random_state=42)
    embedding = model.fit_transform(X)
    return embedding


def run_tsne(X, n_components=2):
    perplexity = min(30.0, X.shape[0] - 1.0)
    model = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    embedding = model.fit_transform(X)
    return embedding


# --- 2. TENSOR ANALYSIS (PARAFAC, TUCKER) ---

def run_parafac(tensor, rank, non_negative=False):
    tensor_no_nan = np.nan_to_num(tensor, nan=0.0)
    if non_negative:
        weights, factors = tl.decomposition.non_negative_parafac(tensor_no_nan, rank=rank, init='random',
                                                                 n_iter_max=100)
    else:
        weights, factors = tl.decomposition.parafac(tensor_no_nan, rank=rank, n_iter_max=100)
    return {'weights': weights, 'factors': factors}


def run_parafac_reconstruction(parafac_results, source_tensor, k):
    weights = parafac_results['weights'][:k]
    factors = [f[:, :k] for f in parafac_results['factors']]
    reconstructed_tensor = tl.cp_to_tensor((weights, factors))
    tensor_no_nan = np.nan_to_num(source_tensor, nan=0.0)
    residual_tensor = tensor_no_nan - reconstructed_tensor
    return {
        'recon_tensor': reconstructed_tensor,
        'residual_tensor': residual_tensor,
        'components': factors[0]
    }


def run_tucker(tensor, ranks):
    tensor_no_nan = np.nan_to_num(tensor, nan=0.0)
    core, factors = tl.decomposition.tucker(tensor_no_nan, rank=ranks)
    return {'core': core, 'factors': factors}


# --- 3. MCR-ALS ---

def run_mcr_als(tensor, rank, non_negative=True, norm=False, closure=False, unimodal=False, use_spexfa_init=True,
                max_iter=500, tol_err_change=1e-4, st_init=None, st_fix_indices=None):
    w, n_rows, m_cols = tensor.shape
    D_matrix_full = tensor.transpose(1, 2, 0).reshape((n_rows * m_cols), w)

    # Odfiltruj wadliwe lub puste komórki
    valid_mask = ~np.isnan(D_matrix_full).any(axis=1)
    zero_mask = np.all(D_matrix_full == 0.0, axis=1)
    valid_mask = valid_mask & ~zero_mask
    D_matrix = D_matrix_full[valid_mask]

    if D_matrix.shape[0] == 0:
        raise ValueError("No valid data for MCR (all NaN or zeros).")

    n_samples, n_features = D_matrix.shape

    c_constraints = []
    st_constraints = []

    if non_negative:
        c_regr = NNLS()
        st_regr = NNLS()
    else:
        c_regr = OLS()
        st_regr = OLS()

    if norm:
        c_constraints.append(ConstraintNorm())
    if closure:
        c_constraints.append(ConstraintClosure(axis=1))
    if unimodal:
        c_constraints.append(ConstraintUnimodal())

    final_st_init = np.random.rand(rank, n_features)

    # Inteligentna inicjalizacja za pomocą SPEXFA
    if use_spexfa_init and st_init is None:
        try:
            spex_res = run_spexfa(D_matrix, rank)
            final_st_init = spex_res['ST'].T
        except Exception as e:
            print(f"SPEXFA init failed: {e}. Fallback to random.")

    if st_init is not None and st_fix_indices:
        for i, fix_idx in enumerate(st_fix_indices):
            if fix_idx < rank:
                final_st_init[fix_idx, :] = st_init[i, :]

    mcr = McrAR(max_iter=max_iter, tol_err_change=tol_err_change,
                c_regr=c_regr, st_regr=st_regr,
                c_constraints=c_constraints, st_constraints=st_constraints)

    mcr.fit(D_matrix, ST=final_st_init, st_fix=st_fix_indices)

    # Odtwarzamy geometrię - przywracamy dziury na siatce macierzy
    C_full = np.full((n_rows * m_cols, rank), np.nan)
    C_full[valid_mask, :] = mcr.C_opt_

    return {
        'C': C_full,
        'ST': mcr.ST_opt_.T,
        'rank': rank
    }


def run_mcr_reconstruction(mcr_results, source_tensor, k):
    C_k = mcr_results['C'][:, :k]
    ST_k = mcr_results['ST'][:, :k]

    D_recon_k = C_k @ ST_k.T
    w = ST_k.shape[0]
    n_rows, m_cols = source_tensor.shape[1], source_tensor.shape[2]

    reconstructed_tensor = D_recon_k.reshape((n_rows, m_cols, w)).transpose(2, 0, 1)

    # NaN pozostaje NaN po odjęciu
    with np.errstate(invalid='ignore'):
        residual_tensor = source_tensor - reconstructed_tensor

    return {
        'recon_tensor': reconstructed_tensor,
        'residual_tensor': residual_tensor,
        'components': ST_k
    }


# --- 4. FACTOR ANALYSIS (MALINOWSKI / SPEXFA) ---

def run_fa_rank_analysis(X):
    r, c = X.shape
    malinowski_r = c
    malinowski_c = r
    sm = min(malinowski_r, malinowski_c)
    max_k = sm - 1

    if max_k < 1: raise ValueError("Not enough data for FA.")

    u, s_vec, vh = np.linalg.svd(X, full_matrices=False)
    ev = s_vec ** 2

    re = np.zeros(max_k)
    ind = np.zeros(max_k)
    sev = np.zeros(sm + 1)

    for k_sev in range(sm - 1, -1, -1):
        sev[k_sev] = sev[k_sev + 1] + ev[k_sev]

    for l in range(max_k):
        matlab_l = l + 1
        current_sev = sev[matlab_l]
        re_val = np.sqrt(current_sev / (malinowski_r * (malinowski_c - matlab_l)))
        re[l] = re_val
        ind[l] = re_val / ((malinowski_c - matlab_l) ** 2)

    return {
        'u': u, 's_vec': s_vec, 'vh': vh,
        'ev': ev[:max_k], 're': re, 'ind': ind, 'max_k': max_k,
        'X_original': X
    }


def run_fa_reconstruction(fa_results, k):
    u = fa_results['u']
    s_vec = fa_results['s_vec']
    vh = fa_results['vh']
    X_original = fa_results['X_original']

    u_k = u[:, :k]
    s_k = np.diag(s_vec[:k])
    vh_k = vh[:k, :]

    X_recon = u_k @ s_k @ vh_k
    residuals = X_original - X_recon

    return {
        'X_reconstructed': X_recon,
        'residuals': residuals,
        'loadings': vh_k.T,
        'scores': u_k @ s_k
    }


def run_spexfa(X, n_components):
    D = X.T
    r, c = D.shape
    n = n_components

    u, s_vec, vh = np.linalg.svd(D, full_matrices=False)
    s_mat = np.diag(s_vec)

    u_n = u[:, :n]
    s_n = s_mat[:n, :n]
    vh_n = vh[:n, :]
    dr = u_n @ s_n @ vh_n

    ev = s_vec ** 2
    sev = np.sum(ev[n:])
    re = np.sqrt(sev / (r * (c - n)))
    cutoff = 5 * re * np.sqrt(n)

    ubar = u_n @ s_n

    ubar_norm = np.linalg.norm(ubar, axis=1)
    mask = ubar_norm < cutoff
    ubar_masked = np.copy(ubar)
    ubar_masked[mask, :] = np.nan

    ubar_norm_clean_indices = ~mask
    if np.sum(ubar_norm_clean_indices) == 0:
        raise ValueError("All variables are below noise threshold.")

    ubar_norm_clean = np.linalg.norm(ubar_masked[ubar_norm_clean_indices, :], axis=1, keepdims=True)
    ubar_masked[ubar_norm_clean_indices, :] = (ubar_masked[ubar_norm_clean_indices, :] * np.sqrt(c)) / ubar_norm_clean

    key = [np.nanargmin(np.abs(ubar_masked[:, 0]))]

    for k_idx in range(1, n):
        w = ubar_masked[:, :k_idx + 1]
        ky = np.zeros((k_idx + 1, k_idx + 1))
        for j in range(k_idx): ky[j, :] = w[key[j], :]

        dt = np.zeros(r)
        for i in range(r):
            if np.isnan(ubar_masked[i, 0]):
                dt[i] = -np.inf
            else:
                ky[k_idx, :] = w[i, :]
                dt[i] = np.abs(np.linalg.det(ky))
        key.append(np.argmax(dt))

    w_all = u_n @ s_n
    iter_count = 0
    while iter_count < 20:
        tkey = key.copy()
        for j in range(n):
            tset = np.zeros((n, n))
            for i in range(n): tset[i, :] = w_all[key[i], :]

            dt = np.zeros(r)
            for k_row in range(r):
                if np.isnan(ubar_masked[k_row, 0]):
                    dt[k_row] = -np.inf
                else:
                    tset[j, :] = w_all[k_row, :]
                    dt[k_row] = np.abs(np.linalg.det(tset))
            key[j] = np.argmax(dt)

        if np.array_equal(key, tkey): break
        iter_count += 1

    conc_transposed = dr[key, :]
    spex = dr @ pinv(conc_transposed)

    return {
        'C': conc_transposed.T,
        'ST': spex,
        'rank': n
    }


# --- 5. 2D-COS ---

def calculate_2dcos(D):
    n_features, n_samples = D.shape
    if n_samples < 2: return None, None

    mean_spec = D.mean(axis=1, keepdims=True)
    Y = D - mean_spec
    Phi = (1.0 / (n_samples - 1)) * (Y @ Y.T)

    N = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j: N[i, j] = 1.0 / (j - i)

    Psi = (1.0 / (n_samples - 1)) * (Y @ N @ Y.T)
    return Phi, Psi