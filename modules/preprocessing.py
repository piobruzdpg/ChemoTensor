# modules/preprocessing.py
import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, eye, spdiags
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LinearRegression

def apply_als_baseline(y, lam, p, n_iter=10):
    """Calculates ALS baseline."""
    L = len(y)
    D = eye(L, format='csc')
    D = D[2:, :] - 2 * D[1:-1, :] + D[:-2, :]
    D = D.T @ D
    w = np.ones(L)
    z = np.zeros(L)
    for i in range(n_iter):
        W = spdiags(w, 0, L, L, format='csc')
        Z = W + lam * D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def correction_als(tensor, lam, p):
    """Applies ALS correction to the whole tensor."""
    tensor_copy = np.copy(tensor)
    w, n, m = tensor_copy.shape
    for r in range(n):
        for c in range(m):
            if not np.isnan(tensor_copy[0, r, c]):
                spectrum = tensor_copy[:, r, c]
                baseline = apply_als_baseline(spectrum, lam, p)
                tensor_copy[:, r, c] = spectrum - baseline
    return tensor_copy

def apply_savgol(tensor, window, poly, deriv=2):
    """Savitzky-Golay Filter."""
    tensor_copy = np.copy(tensor)
    w, n, m = tensor_copy.shape
    for r in range(n):
        for c in range(m):
            if not np.isnan(tensor_copy[0, r, c]):
                tensor_copy[:, r, c] = savgol_filter(tensor_copy[:, r, c], window, poly, deriv=deriv)
    return tensor_copy

def apply_snv(tensor):
    """Standard Normal Variate (SNV)."""
    tensor_copy = np.copy(tensor)
    w, n, m = tensor_copy.shape
    for r in range(n):
        for c in range(m):
            if not np.isnan(tensor_copy[0, r, c]):
                spectrum = tensor_copy[:, r, c]
                mean = np.mean(spectrum)
                std = np.std(spectrum)
                if std > 1e-8:
                    tensor_copy[:, r, c] = (spectrum - mean) / std
                else:
                    tensor_copy[:, r, c] = spectrum - mean
    return tensor_copy

def apply_min_max(tensor):
    """Min-Max Scaling (0-1)."""
    tensor_copy = np.copy(tensor)
    w, n, m = tensor_copy.shape
    for r in range(n):
        for c in range(m):
            if not np.isnan(tensor_copy[0, r, c]):
                spectrum = tensor_copy[:, r, c]
                min_val = np.min(spectrum)
                range_val = np.max(spectrum) - min_val
                if range_val > 1e-8:
                    tensor_copy[:, r, c] = (spectrum - min_val) / range_val
                else:
                    tensor_copy[:, r, c] = 0.5
    return tensor_copy

def apply_l1_norm(tensor):
    """Area Normalization (L1)."""
    tensor_copy = np.copy(tensor)
    w, n, m = tensor_copy.shape
    for r in range(n):
        for c in range(m):
            if not np.isnan(tensor_copy[0, r, c]):
                spectrum = tensor_copy[:, r, c]
                area = np.sum(np.abs(spectrum))
                if area > 1e-8:
                    tensor_copy[:, r, c] = spectrum / area
    return tensor_copy

def apply_msc(tensor):
    """Multiplicative Scatter Correction (MSC)."""
    tensor_copy = np.copy(tensor)
    w, n, m = tensor_copy.shape
    
    # 1. Calculate mean spectrum
    all_spectra = []
    for r in range(n):
        for c in range(m):
            if not np.isnan(tensor_copy[0, r, c]):
                all_spectra.append(tensor_copy[:, r, c])
    
    if not all_spectra:
        raise ValueError("No data to calculate reference spectrum.")

    mean_spectrum = np.mean(np.array(all_spectra), axis=0).reshape(-1, 1)
    model = LinearRegression()

    # 2. Fit and correct
    for r in range(n):
        for c in range(m):
            if not np.isnan(tensor_copy[0, r, c]):
                spectrum = tensor_copy[:, r, c].reshape(-1, 1)
                model.fit(mean_spectrum, spectrum)
                intercept = model.intercept_[0]
                slope = model.coef_[0][0]

                if np.abs(slope) > 1e-8:
                    tensor_copy[:, r, c] = (tensor_copy[:, r, c] - intercept) / slope
                else:
                    tensor_copy[:, r, c] = tensor_copy[:, r, c] - intercept
    return tensor_copy

def apply_offset(tensor, wavenumbers, target_wavenumber, target_value=0.0):
    """Przesuwa widma tak, aby w punkcie target_wavenumber miały wartość target_value."""
    tensor_copy = np.copy(tensor)
    w, n, m = tensor_copy.shape
    
    # Znajdź indeks najbliższy podanej liczbie falowej
    if wavenumbers is None:
        raise ValueError("Wavenumbers axis is missing.")
        
    idx = (np.abs(wavenumbers - target_wavenumber)).argmin()
    
    # Sprawdź, czy nie jesteśmy "poza zakresem" (opcjonalne, ale dobre dla bezpieczeństwa)
    # Tutaj ufamy argmin, że znajdzie najbliższy.
    
    for r in range(n):
        for c in range(m):
            if not np.isnan(tensor_copy[0, r, c]):
                spectrum = tensor_copy[:, r, c]
                current_val = spectrum[idx]
                shift = target_value - current_val
                tensor_copy[:, r, c] = spectrum + shift
    return tensor_copy