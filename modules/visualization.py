# modules/visualization.py
import matplotlib.figure
import matplotlib.pyplot
import numpy as np

# --- Paleta kolorów ---
THEME_COLORS = {
    'Light': {'bg_color': '#ffffff', 'text_color': '#000000', 'grid_color': '#cccccc', 'spine_color': '#bbbbbb'},
    'Dark': {'bg_color': '#2b2b2b', 'text_color': '#ffffff', 'grid_color': '#555555', 'spine_color': '#777777'}
}

def apply_theme(ax, theme_name='Light'):
    theme = THEME_COLORS[theme_name]
    ax.set_facecolor(theme['bg_color'])
    ax.xaxis.label.set_color(theme['text_color'])
    ax.yaxis.label.set_color(theme['text_color'])
    ax.title.set_color(theme['text_color'])
    ax.tick_params(axis='x', colors=theme['text_color'])
    ax.tick_params(axis='y', colors=theme['text_color'])
    for spine in ax.spines.values():
        spine.set_color(theme['spine_color'])
    ax.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

def invert_xaxis_if_wavenumbers(ax, wavenumbers):
    """Odwraca oś X dla spektroskopii (liczby falowe)."""
    if wavenumbers is not None and len(wavenumbers) > 0:
        # Sprawdzamy czy już nie jest odwrócona, żeby nie machać w kółko
        current_xlim = ax.get_xlim()
        if current_xlim[0] < current_xlim[1]: # Jeżeli rośnie w prawo (standardowo) -> odwróć
             ax.set_xlim(np.max(wavenumbers), np.min(wavenumbers))

def plot_spectra(ax, wavenumbers, data, selected_coords, title_suffix=""):
    """Rysuje zwykłe widma (JEDEN DUŻY PANEL)."""
    # Nie musimy już ukrywać axes[0,1] itd., bo ich nie będzie
    ax.set_visible(True)
    apply_theme(ax)
    
    if not selected_coords:
        ax.set_title("Panel Wykresów (Wybierz komórki)")
        return

    ax.set_title(f"Zaznaczono {len(selected_coords)} widm {title_suffix}")
    ax.set_xlabel("Liczba falowa")
    ax.set_ylabel("Intensywność")

    if data is not None and wavenumbers is not None:
        for coords in selected_coords:
            r, c = coords
            if not np.isnan(data[0, r, c]):
                ax.plot(wavenumbers, data[:, r, c], label=f"({r},{c})")
        
        invert_xaxis_if_wavenumbers(ax, wavenumbers)
        if len(selected_coords) <= 10:
            ax.legend(loc='best')

def plot_pca(axes, results, wavenumbers):
    """Rysuje wyniki PCA."""
    for ax in axes.flat: 
        ax.set_visible(True)
        apply_theme(ax)

    scores = results['scores']
    loadings = results['loadings']
    variance = results['variance']
    labels = results['labels']
    n_components = scores.shape[1]

    # 1. Score Plot
    ax1 = axes[0, 0]
    ax1.scatter(scores[:, 0], scores[:, 1])
    for i, label in enumerate(labels):
        ax1.text(scores[i, 0], scores[i, 1], label, fontsize=9)
    ax1.set_xlabel(f"PC1 ({variance[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({variance[1]*100:.1f}%)")
    ax1.set_title("PCA: Score Plot")

    # 2. Loadings
    ax2 = axes[0, 1]
    if wavenumbers is not None:
        for i in range(n_components):
            ax2.plot(wavenumbers, loadings[:, i], label=f"PC{i+1}")
    ax2.set_title("PCA: Loadings")
    invert_xaxis_if_wavenumbers(ax2, wavenumbers)
    ax2.legend()

    # 3. Scores vs Sample
    ax3 = axes[1, 0]
    sample_indices = np.arange(len(labels))
    for i in range(n_components):
        ax3.plot(sample_indices, scores[:, i], 'o-', markersize=4, label=f'PC{i+1}')
    ax3.set_title("Scores vs Sample")
    ax3.set_xticks(sample_indices)
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # 4. Scree Plot
    ax4 = axes[1, 1]
    pc_range = np.arange(1, len(variance) + 1)
    ax4.bar(pc_range, variance * 100, alpha=0.7)
    ax4.plot(pc_range, np.cumsum(variance * 100), 'r-o')
    ax4.set_title("Scree Plot")
    ax4.set_ylabel("Wariancja (%)")

def plot_mcr(axes, results, wavenumbers, n_rows, m_cols):
    """Rysuje wyniki MCR."""
    for ax in axes.flat: 
        ax.set_visible(True)
        apply_theme(ax)

    C = results['C']
    ST = results['ST']
    k = results['rank']

    # 1. Widma
    ax1 = axes[0, 0]
    if wavenumbers is not None:
        for i in range(k): 
            ax1.plot(wavenumbers, ST[:, i], label=f"Skł. {i+1}")
    ax1.set_title("MCR: Widma")
    invert_xaxis_if_wavenumbers(ax1, wavenumbers)
    ax1.legend()

    # 2. Stężenia liniowo
    ax2 = axes[0, 1]
    x_range = np.arange(C.shape[0])
    for i in range(k):
        ax2.plot(x_range, C[:, i], label=f"Skł. {i+1}")
    ax2.set_title("MCR: Stężenia")
    ax2.legend()

    # 3. Mapa Składnika 1
    ax3 = axes[1, 0]
    c1_map = C[:, 0].reshape((n_rows, m_cols))
    im3 = ax3.imshow(c1_map, aspect='auto', interpolation='nearest', cmap='viridis')
    ax3.set_title("MCR: Mapa Skł. 1")
    matplotlib.pyplot.colorbar(im3, ax=ax3)

    # 4. Mapa Składnika 2 (jeśli istnieje)
    ax4 = axes[1, 1]
    if k > 1:
        c2_map = C[:, 1].reshape((n_rows, m_cols))
        im4 = ax4.imshow(c2_map, aspect='auto', interpolation='nearest', cmap='viridis')
        ax4.set_title("MCR: Mapa Skł. 2")
        matplotlib.pyplot.colorbar(im4, ax=ax4)
    else:
        ax4.set_visible(False)

def plot_heatmap(ax, data, wavenumbers, n_rows, m_cols):
    """Rysuje heatmapę (JEDEN DUŻY PANEL)."""
    ax.set_visible(True)
    apply_theme(ax)

    if data is not None:
        w, n, m = data.shape
        data_2d = np.nan_to_num(data, nan=0.0).transpose(0, 2, 1).reshape(w, n * m)
        
        v_min, v_max = np.min(data_2d), np.max(data_2d)
        im = ax.pcolormesh(data_2d, cmap='viridis', vmin=v_min, vmax=v_max)
        
        ax.set_title("Heatmapa")
        ax.set_ylabel("Liczba falowa (Indeks)")
        ax.set_xlabel("Próbka")
        ax.invert_yaxis()
        matplotlib.pyplot.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "Brak danych", ha='center', va='center')

def plot_manifold(ax, results):
    """Dla UMAP / t-SNE (JEDEN DUŻY PANEL)."""
    ax.set_visible(True)
    apply_theme(ax)

    scores = results['scores']
    labels = results['labels']
    method = results['method_name']

    ax.scatter(scores[:, 0], scores[:, 1])
    for i, label in enumerate(labels):
        ax.text(scores[i, 0], scores[i, 1], label, fontsize=9)
    ax.set_title(f"Wynik {method}")

def plot_2dcos(axes, results, wavenumbers, slice_idx):
    """Rysuje mapy 2D-COS (Synchroniczna i Asynchroniczna)."""
    # Ukrywamy dolne wykresy, używamy tylko górnych
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    axes[1, 0].set_visible(False)
    axes[1, 1].set_visible(False)
    
    apply_theme(ax1)
    apply_theme(ax2)

    phi_map = results['phi'][:, :, slice_idx]
    psi_map = results['psi'][:, :, slice_idx]

    # Ustalanie zakresu osi (extent)
    if wavenumbers is not None:
        v_min_ax = np.min(wavenumbers)
        v_max_ax = np.max(wavenumbers)
        # Matplotlib extent: [left, right, bottom, top]
        # Dla liczb falowych malejących: [max, min, max, min]
        extent = [v_max_ax, v_min_ax, v_max_ax, v_min_ax]
    else:
        extent = None

    # Synchroniczna
    vmax_phi = np.max(np.abs(phi_map[np.isfinite(phi_map)]))
    im1 = ax1.imshow(phi_map, cmap='RdBu_r', vmin=-vmax_phi, vmax=vmax_phi, 
                     extent=extent, interpolation='nearest')
    ax1.set_title(f"Mapa Synchroniczna (Plaster {slice_idx})")
    ax1.set_xlabel("v1")
    ax1.set_ylabel("v2")
    matplotlib.pyplot.colorbar(im1, ax=ax1)

    # Asynchroniczna
    vmax_psi = np.max(np.abs(psi_map[np.isfinite(psi_map)]))
    im2 = ax2.imshow(psi_map, cmap='RdBu_r', vmin=-vmax_psi, vmax=vmax_psi, 
                     extent=extent, interpolation='nearest')
    ax2.set_title(f"Mapa Asynchroniczna (Plaster {slice_idx})")
    ax2.set_xlabel("v1")
    ax2.set_ylabel("v2")
    matplotlib.pyplot.colorbar(im2, ax=ax2)

def plot_pls(ax, results, wavenumbers):
    """Rysuje wyniki PLS (JEDEN DUŻY PANEL)."""
    ax.set_visible(True)
    apply_theme(ax)

    coefs = results['coefs']
    
    if wavenumbers is not None:
        ax.plot(wavenumbers, coefs)
        invert_xaxis_if_wavenumbers(ax, wavenumbers)
        ax.set_xlabel("Liczba falowa")
    else:
        ax.plot(coefs)
        ax.set_xlabel("Indeks Zmiennej")

    ax.set_ylabel("Współczynniki Regresji")
    ax.set_title("Ważność Zmiennych (PLS Coefficients)")
    ax.axhline(0, color='gray', linestyle='--')

# --- modules/visualization.py (DOPISZ NA KOŃCU) ---

def plot_tucker(axes, results, wavenumbers, n_rows, m_cols):
    """Wizualizacja wyników Tuckera."""
    for ax in axes.flat: 
        ax.set_visible(True)
        apply_theme(ax)

    core = results['core']       # (r_w, r_n, r_m)
    factors = results['factors'] # [(w, rw), (n, rn), (m, rm)]
    
    # 1. Widma (Mode 0)
    ax1 = axes[0, 0]
    r_w = factors[0].shape[1]
    if wavenumbers is not None:
        for i in range(r_w):
            ax1.plot(wavenumbers, factors[0][:, i], label=f"Skł. {i+1}")
    ax1.set_title("Tucker: Widma (Mode 0)")
    invert_xaxis_if_wavenumbers(ax1, wavenumbers)
    if r_w <= 10: ax1.legend()

    # 2. Trendy Wierszy (Mode 1)
    ax2 = axes[0, 1]
    r_n = factors[1].shape[1]
    for i in range(r_n):
        ax2.plot(factors[1][:, i], 'o-', label=f"Skł. {i+1}")
    ax2.set_title("Tucker: Trendy Wierszy (Mode 1)")
    if r_n <= 10: ax2.legend()

    # 3. Trendy Kolumn (Mode 2)
    ax3 = axes[1, 0]
    r_m = factors[2].shape[1]
    for i in range(r_m):
        ax3.plot(factors[2][:, i], 'o-', label=f"Skł. {i+1}")
    ax3.set_title("Tucker: Trendy Kolumn (Mode 2)")
    if r_m <= 10: ax3.legend()

    # 4. Rdzeń (Core) - Plaster 0
    ax4 = axes[1, 1]
    core_slice = core[:, :, 0] # (rw, rn) dla rm=0
    vmax = np.max(np.abs(core_slice))
    im = ax4.imshow(core_slice, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax4.set_title("Tensor Rdzeniowy (Plaster 0)")
    ax4.set_ylabel("Widma (Mode 0)")
    ax4.set_xlabel("Wiersze (Mode 1)")
    matplotlib.pyplot.colorbar(im, ax=ax4)

def plot_fa_rank(axes, results):
    """Wizualizacja analizy rangi (Malinowski)."""
    for ax in axes.flat: ax.set_visible(True)
    apply_theme(axes[0,0]); apply_theme(axes[0,1]); apply_theme(axes[1,0]); apply_theme(axes[1,1])

    ev = results['ev']
    re = results['re']
    ind = results['ind']
    max_k = results['max_k']
    x_ax = np.arange(1, max_k + 1)

    # 1. Eigenvalues
    axes[0, 0].plot(x_ax, ev, 'o-')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title("Wartości Własne (Log)")
    axes[0, 0].set_xlabel("Faktor")

    # 2. RE & IND
    ax2 = axes[0, 1]
    ax2.plot(x_ax, np.log(re), 'b-o', label='log(RE)')
    ax2.set_ylabel('log(RE)', color='b')
    ax2.set_title("Wskaźniki Błędu (Malinowski)")
    
    ax2b = ax2.twinx()
    ax2b.plot(x_ax, ind, 'r-o', label='IND')
    ax2b.set_ylabel('IND', color='r')
    
    # 3. Widma Abstrakcyjne (vh)
    axes[1, 0].plot(results['vh'][:max_k].T)
    axes[1, 0].set_title("Widma Abstrakcyjne (Loadings)")
    
    # 4. Profile Abstrakcyjne (u*s)
    scores = results['u'][:, :max_k] @ np.diag(results['s_vec'][:max_k])
    axes[1, 1].plot(scores)
    axes[1, 1].set_title("Profile Abstrakcyjne (Scores)")

def plot_reconstruction(axes, results, wavenumbers, selected_coords=None):
    """Wizualizacja rekonstrukcji (PCA, FA, PARAFAC, MCR)."""
    for ax in axes.flat: ax.set_visible(True)
    for ax in axes.flat: apply_theme(ax)

    # Rozpakowanie (obsługuje różne źródła, jeśli klucze są spójne w analysis.py)
    if 'X_reconstructed' in results:
        recon = results['X_reconstructed']
        resid = results['residuals']
        orig = results.get('X_original') # Dla PCA/FA
    else:
        # Dla Tensorów (PARAFAC/MCR)
        recon = results['recon_tensor']
        resid = results['residual_tensor']
        orig = results.get('source_tensor') # Musi być przekazane

    # Jeśli tensor 3D, musimy wybrać co rysować (np. zaznaczone)
    # Dla uproszczenia rysujemy spłaszczone lub wybrane
    
    # 1. Widma Oryginalne (podgląd)
    ax1 = axes[0, 1]
    ax1.set_title("Oryginał")
    
    # 2. Widma Odtworzone
    ax2 = axes[1, 0]
    ax2.set_title("Odtworzone")
    
    # 3. Rezydua
    ax3 = axes[1, 1]
    ax3.set_title("Rezydua (Różnica)")
    
    # Rysowanie (logika zależy od tego czy mamy macierz czy tensor)
    if recon.ndim == 3: # Tensor
        # Rysujemy tylko zaznaczone punkty, jeśli są, albo średnią
        targets = selected_coords if selected_coords else [(0,0)]
        for r, c in targets:
            if wavenumbers is not None:
                if orig is not None: ax1.plot(wavenumbers, orig[:, r, c], alpha=0.5)
                ax2.plot(wavenumbers, recon[:, r, c], alpha=0.5)
                ax3.plot(wavenumbers, resid[:, r, c], alpha=0.5)
    else: # Macierz (PCA/FA)
        if wavenumbers is not None:
            if orig is not None: ax1.plot(wavenumbers, orig.T, alpha=0.3)
            ax2.plot(wavenumbers, recon.T, alpha=0.3)
            ax3.plot(wavenumbers, resid.T, alpha=0.3)
    
    if wavenumbers is not None:
        invert_xaxis_if_wavenumbers(ax1, wavenumbers)
        invert_xaxis_if_wavenumbers(ax2, wavenumbers)
        invert_xaxis_if_wavenumbers(ax3, wavenumbers)

    # 4. Ładunki / Komponenty użyte
    ax4 = axes[0, 0]
    if 'components' in results: # Tensor
        comps = results['components']
        ax4.plot(wavenumbers, comps)
    elif 'loadings' in results: # PCA
        # loadings w PCA są (n_features, n_components)
        ax4.plot(wavenumbers, results['loadings'])
        
    ax4.set_title(f"Użyte Składowe")
    if wavenumbers is not None: invert_xaxis_if_wavenumbers(ax4, wavenumbers)

# --- modules/visualization.py (DOPISZ NA KOŃCU) ---

def plot_spexfa(axes, results, wavenumbers):
    """Wizualizacja SPEXFA (bez mapowania 2D, dla dowolnej liczby próbek)."""
    for ax in axes.flat: 
        ax.set_visible(True)
        apply_theme(ax)

    C = results['C']    # Stężenia (n_samples, k)
    ST = results['ST']  # Widma (n_features, k)
    k = results['rank']
    labels = results.get('labels', []) # Etykiety, np. ["(0,0)", "(1,2)"]
    n_samples = C.shape[0]
    x_range = np.arange(n_samples)

    # 1. Wyizolowane Widma (Top Left)
    ax1 = axes[0, 0]
    if wavenumbers is not None:
        for i in range(k):
            ax1.plot(wavenumbers, ST[:, i], label=f"Faktor {i+1}")
    ax1.set_title("SPEXFA: Wyizolowane Widma")
    ax1.set_xlabel("Liczba falowa")
    invert_xaxis_if_wavenumbers(ax1, wavenumbers)
    if k <= 10: ax1.legend()

    # 2. Profile Stężeń - Liniowy (Top Right)
    ax2 = axes[0, 1]
    for i in range(k):
        ax2.plot(x_range, C[:, i], 'o-', label=f"Faktor {i+1}", alpha=0.7)
    ax2.set_title("Profile Stężeń (Wszystkie)")
    ax2.set_xticks(x_range)
    if len(labels) == n_samples:
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # 3. Stężenia Faktor 1 - Słupkowy (Bottom Left)
    ax3 = axes[1, 0]
    ax3.bar(x_range, C[:, 0], color='#3498DB', alpha=0.8)
    ax3.set_title("Udział: Faktor 1")
    ax3.set_xticks(x_range)
    if len(labels) == n_samples:
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    # 4. Stężenia Faktor 2 - Słupkowy (Bottom Right)
    ax4 = axes[1, 1]
    if k > 1:
        ax4.bar(x_range, C[:, 1], color='#E74C3C', alpha=0.8)
        ax4.set_title("Udział: Faktor 2")
        ax4.set_xticks(x_range)
        if len(labels) == n_samples:
            ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    else:
        ax4.text(0.5, 0.5, "Brak drugiego faktora", ha='center', va='center')