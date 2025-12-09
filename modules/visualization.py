# modules/visualization.py
import matplotlib.figure
import matplotlib.pyplot
import numpy as np

# --- Color Theme ---
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
    """Inverts X-axis for spectroscopy (wavenumbers)."""
    if wavenumbers is not None and len(wavenumbers) > 0:
        # Check if not already inverted
        current_xlim = ax.get_xlim()
        if current_xlim[0] < current_xlim[1]: # If increasing to the right (standard) -> invert
             ax.set_xlim(np.max(wavenumbers), np.min(wavenumbers))

def plot_spectra(ax, wavenumbers, data, selected_coords, title_suffix=""):
    """Plots standard spectra (SINGLE LARGE PANEL)."""
    ax.set_visible(True)
    apply_theme(ax)
    
    if not selected_coords:
        ax.set_title("Plot Panel (Select cells)")
        return

    ax.set_title(f"Selected {len(selected_coords)} spectra {title_suffix}")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Intensity")

    if data is not None and wavenumbers is not None:
        for coords in selected_coords:
            r, c = coords
            if not np.isnan(data[0, r, c]):
                ax.plot(wavenumbers, data[:, r, c], label=f"({r},{c})")
        
        invert_xaxis_if_wavenumbers(ax, wavenumbers)
        if len(selected_coords) <= 10:
            ax.legend(loc='best')

def plot_pca(axes, results, wavenumbers):
    """Plots PCA results."""
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
    ax4.set_ylabel("Variance (%)")

def plot_mcr(axes, results, wavenumbers, n_rows, m_cols):
    """Plots MCR results."""
    for ax in axes.flat: 
        ax.set_visible(True)
        apply_theme(ax)

    C = results['C']
    ST = results['ST']
    k = results['rank']

    # 1. Spectra
    ax1 = axes[0, 0]
    if wavenumbers is not None:
        for i in range(k): 
            ax1.plot(wavenumbers, ST[:, i], label=f"Comp. {i+1}")
    ax1.set_title("MCR: Spectra")
    invert_xaxis_if_wavenumbers(ax1, wavenumbers)
    ax1.legend()

    # 2. Concentrations (Linear)
    ax2 = axes[0, 1]
    x_range = np.arange(C.shape[0])
    for i in range(k):
        ax2.plot(x_range, C[:, i], label=f"Comp. {i+1}")
    ax2.set_title("MCR: Concentrations")
    ax2.legend()

    # 3. Map Component 1
    ax3 = axes[1, 0]
    c1_map = C[:, 0].reshape((n_rows, m_cols))
    im3 = ax3.imshow(c1_map, aspect='auto', interpolation='nearest', cmap='viridis')
    ax3.set_title("MCR: Map Comp. 1")
    matplotlib.pyplot.colorbar(im3, ax=ax3)

    # 4. Map Component 2 (if exists)
    ax4 = axes[1, 1]
    if k > 1:
        c2_map = C[:, 1].reshape((n_rows, m_cols))
        im4 = ax4.imshow(c2_map, aspect='auto', interpolation='nearest', cmap='viridis')
        ax4.set_title("MCR: Map Comp. 2")
        matplotlib.pyplot.colorbar(im4, ax=ax4)
    else:
        ax4.set_visible(False)

def plot_heatmap(ax, data, wavenumbers, n_rows, m_cols):
    """Plots Heatmap (SINGLE LARGE PANEL)."""
    ax.set_visible(True)
    apply_theme(ax)

    if data is not None:
        w, n, m = data.shape
        data_2d = np.nan_to_num(data, nan=0.0).transpose(0, 2, 1).reshape(w, n * m)
        
        v_min, v_max = np.min(data_2d), np.max(data_2d)
        im = ax.pcolormesh(data_2d, cmap='viridis', vmin=v_min, vmax=v_max)
        
        ax.set_title("Heatmap")
        ax.set_ylabel("Wavenumber (Index)")
        ax.set_xlabel("Sample")
        ax.invert_yaxis()
        matplotlib.pyplot.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')

def plot_manifold(ax, results):
    """For UMAP / t-SNE (SINGLE LARGE PANEL)."""
    ax.set_visible(True)
    apply_theme(ax)

    scores = results['scores']
    labels = results['labels']
    method = results['method_name']

    ax.scatter(scores[:, 0], scores[:, 1])
    for i, label in enumerate(labels):
        ax.text(scores[i, 0], scores[i, 1], label, fontsize=9)
    ax.set_title(f"{method} Result")

def plot_2dcos(axes, results, wavenumbers, slice_idx):
    """Plots 2D-COS Maps (Synchronous and Asynchronous)."""
    # Hide bottom plots, use only top ones
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    axes[1, 0].set_visible(False)
    axes[1, 1].set_visible(False)
    
    apply_theme(ax1)
    apply_theme(ax2)

    phi_map = results['phi'][:, :, slice_idx]
    psi_map = results['psi'][:, :, slice_idx]

    # Extent configuration
    if wavenumbers is not None:
        v_min_ax = np.min(wavenumbers)
        v_max_ax = np.max(wavenumbers)
        # Matplotlib extent: [left, right, bottom, top]
        # For decreasing wavenumbers: [max, min, max, min]
        extent = [v_max_ax, v_min_ax, v_max_ax, v_min_ax]
    else:
        extent = None

    # Synchronous
    vmax_phi = np.max(np.abs(phi_map[np.isfinite(phi_map)]))
    im1 = ax1.imshow(phi_map, cmap='RdBu_r', vmin=-vmax_phi, vmax=vmax_phi, 
                     extent=extent, interpolation='nearest')
    ax1.set_title(f"Synchronous Map (Slice {slice_idx})")
    ax1.set_xlabel("v1")
    ax1.set_ylabel("v2")
    matplotlib.pyplot.colorbar(im1, ax=ax1)

    # Asynchronous
    vmax_psi = np.max(np.abs(psi_map[np.isfinite(psi_map)]))
    im2 = ax2.imshow(psi_map, cmap='RdBu_r', vmin=-vmax_psi, vmax=vmax_psi, 
                     extent=extent, interpolation='nearest')
    ax2.set_title(f"Asynchronous Map (Slice {slice_idx})")
    ax2.set_xlabel("v1")
    ax2.set_ylabel("v2")
    matplotlib.pyplot.colorbar(im2, ax=ax2)

def plot_pls(ax, results, wavenumbers):
    """Plots PLS results (SINGLE LARGE PANEL)."""
    ax.set_visible(True)
    apply_theme(ax)

    coefs = results['coefs']
    
    if wavenumbers is not None:
        ax.plot(wavenumbers, coefs)
        invert_xaxis_if_wavenumbers(ax, wavenumbers)
        ax.set_xlabel("Wavenumber")
    else:
        ax.plot(coefs)
        ax.set_xlabel("Variable Index")

    ax.set_ylabel("Regression Coefficients")
    ax.set_title("Variable Importance (PLS Coefficients)")
    ax.axhline(0, color='gray', linestyle='--')

def plot_tucker(axes, results, wavenumbers, n_rows, m_cols):
    """Visualizes Tucker results."""
    for ax in axes.flat: 
        ax.set_visible(True)
        apply_theme(ax)

    core = results['core']       # (r_w, r_n, r_m)
    factors = results['factors'] # [(w, rw), (n, rn), (m, rm)]
    
    # 1. Spectra (Mode 0)
    ax1 = axes[0, 0]
    r_w = factors[0].shape[1]
    if wavenumbers is not None:
        for i in range(r_w):
            ax1.plot(wavenumbers, factors[0][:, i], label=f"Comp. {i+1}")
    ax1.set_title("Tucker: Spectra (Mode 0)")
    invert_xaxis_if_wavenumbers(ax1, wavenumbers)
    if r_w <= 10: ax1.legend()

    # 2. Row Trends (Mode 1)
    ax2 = axes[0, 1]
    r_n = factors[1].shape[1]
    for i in range(r_n):
        ax2.plot(factors[1][:, i], 'o-', label=f"Comp. {i+1}")
    ax2.set_title("Tucker: Row Trends (Mode 1)")
    if r_n <= 10: ax2.legend()

    # 3. Column Trends (Mode 2)
    ax3 = axes[1, 0]
    r_m = factors[2].shape[1]
    for i in range(r_m):
        ax3.plot(factors[2][:, i], 'o-', label=f"Comp. {i+1}")
    ax3.set_title("Tucker: Column Trends (Mode 2)")
    if r_m <= 10: ax3.legend()

    # 4. Core Tensor - Slice 0
    ax4 = axes[1, 1]
    core_slice = core[:, :, 0] # (rw, rn) for rm=0
    vmax = np.max(np.abs(core_slice))
    im = ax4.imshow(core_slice, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax4.set_title("Core Tensor (Slice 0)")
    ax4.set_ylabel("Spectra (Mode 0)")
    ax4.set_xlabel("Rows (Mode 1)")
    matplotlib.pyplot.colorbar(im, ax=ax4)

def plot_fa_rank(axes, results):
    """Visualizes Rank Analysis (Malinowski)."""
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
    axes[0, 0].set_title("Eigenvalues (Log)")
    axes[0, 0].set_xlabel("Factor")

    # 2. RE & IND
    ax2 = axes[0, 1]
    ax2.plot(x_ax, np.log(re), 'b-o', label='log(RE)')
    ax2.set_ylabel('log(RE)', color='b')
    ax2.set_title("Error Indicators (Malinowski)")
    
    ax2b = ax2.twinx()
    ax2b.plot(x_ax, ind, 'r-o', label='IND')
    ax2b.set_ylabel('IND', color='r')
    
    # 3. Abstract Spectra (Loadings)
    axes[1, 0].plot(results['vh'][:max_k].T)
    axes[1, 0].set_title("Abstract Spectra (Loadings)")
    
    # 4. Abstract Profiles (Scores)
    scores = results['u'][:, :max_k] @ np.diag(results['s_vec'][:max_k])
    axes[1, 1].plot(scores)
    axes[1, 1].set_title("Abstract Profiles (Scores)")

def plot_reconstruction(axes, results, wavenumbers, selected_coords=None):
    """Visualizes reconstruction (PCA, FA, PARAFAC, MCR)."""
    for ax in axes.flat: ax.set_visible(True)
    for ax in axes.flat: apply_theme(ax)

    if 'X_reconstructed' in results:
        recon = results['X_reconstructed']
        resid = results['residuals']
        orig = results.get('X_original') # For PCA/FA
    else:
        # For Tensors (PARAFAC/MCR)
        recon = results['recon_tensor']
        resid = results['residual_tensor']
        orig = results.get('source_tensor')

    # 1. Original Spectra
    ax1 = axes[0, 1]
    ax1.set_title("Original")
    
    # 2. Reconstructed Spectra
    ax2 = axes[1, 0]
    ax2.set_title("Reconstructed")
    
    # 3. Residuals
    ax3 = axes[1, 1]
    ax3.set_title("Residuals (Difference)")
    
    # Drawing logic
    if recon.ndim == 3: # Tensor
        targets = selected_coords if selected_coords else [(0,0)]
        for r, c in targets:
            if wavenumbers is not None:
                if orig is not None: ax1.plot(wavenumbers, orig[:, r, c], alpha=0.5)
                ax2.plot(wavenumbers, recon[:, r, c], alpha=0.5)
                ax3.plot(wavenumbers, resid[:, r, c], alpha=0.5)
    else: # Matrix (PCA/FA)
        if wavenumbers is not None:
            if orig is not None: ax1.plot(wavenumbers, orig.T, alpha=0.3)
            ax2.plot(wavenumbers, recon.T, alpha=0.3)
            ax3.plot(wavenumbers, resid.T, alpha=0.3)
    
    if wavenumbers is not None:
        invert_xaxis_if_wavenumbers(ax1, wavenumbers)
        invert_xaxis_if_wavenumbers(ax2, wavenumbers)
        invert_xaxis_if_wavenumbers(ax3, wavenumbers)

    # 4. Used Components
    ax4 = axes[0, 0]
    if 'components' in results: # Tensor
        comps = results['components']
        ax4.plot(wavenumbers, comps)
    elif 'loadings' in results: # PCA
        ax4.plot(wavenumbers, results['loadings'])
        
    ax4.set_title(f"Used Components")
    if wavenumbers is not None: invert_xaxis_if_wavenumbers(ax4, wavenumbers)

def plot_spexfa(axes, results, wavenumbers):
    """Visualizes SPEXFA."""
    for ax in axes.flat: 
        ax.set_visible(True)
        apply_theme(ax)

    C = results['C']    # Concentrations
    ST = results['ST']  # Spectra
    k = results['rank']
    labels = results.get('labels', [])
    n_samples = C.shape[0]
    x_range = np.arange(n_samples)

    # 1. Isolated Spectra (Top Left)
    ax1 = axes[0, 0]
    if wavenumbers is not None:
        for i in range(k):
            ax1.plot(wavenumbers, ST[:, i], label=f"Factor {i+1}")
    ax1.set_title("SPEXFA: Isolated Spectra")
    ax1.set_xlabel("Wavenumber")
    invert_xaxis_if_wavenumbers(ax1, wavenumbers)
    if k <= 10: ax1.legend()

    # 2. Concentration Profiles - Linear (Top Right)
    ax2 = axes[0, 1]
    for i in range(k):
        ax2.plot(x_range, C[:, i], 'o-', label=f"Factor {i+1}", alpha=0.7)
    ax2.set_title("Concentration Profiles (All)")
    ax2.set_xticks(x_range)
    if len(labels) == n_samples:
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # 3. Factor 1 Contribution (Bottom Left)
    ax3 = axes[1, 0]
    ax3.bar(x_range, C[:, 0], color='#3498DB', alpha=0.8)
    ax3.set_title("Contribution: Factor 1")
    ax3.set_xticks(x_range)
    if len(labels) == n_samples:
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    # 4. Factor 2 Contribution (Bottom Right)
    ax4 = axes[1, 1]
    if k > 1:
        ax4.bar(x_range, C[:, 1], color='#E74C3C', alpha=0.8)
        ax4.set_title("Contribution: Factor 2")
        ax4.set_xticks(x_range)
        if len(labels) == n_samples:
            ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No second factor", ha='center', va='center')