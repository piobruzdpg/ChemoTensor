# modules/data_io.py
import json
import pandas as pd
import numpy as np
import openpyxl

def load_single_csv(file_path):
    """Wczytuje pojedynczy plik CSV i zwraca liczby falowe oraz absorbancję."""
    try:
        data = pd.read_csv(file_path, header=None, usecols=[0, 1]).values
        return data[:, 0], data[:, 1]
    except Exception as e:
        raise IOError(f"Błąd odczytu pliku: {e}")

# --- modules/data_io.py ---

def save_project_json(filepath, n_rows, m_cols, wavenumbers, tensor_data, status_map, pipeline_steps):
    """
    Zapisuje stan projektu do JSON, w tym historię preprocessingu.
    """
    project_data = {
        'version': '2.0', # Warto wersjonować pliki
        'n_rows': n_rows,
        'm_cols': m_cols,
        # Konwersja numpy array na listę dla JSON
        'original_wavenumbers': wavenumbers.tolist() if wavenumbers is not None else None,
        # Zapisujemy tylko dane, gdzie nie ma NaN (opcjonalna optymalizacja) lub całość
        # Tutaj zapisujemy całość z obsługą None dla JSON
        'original_tensor_data': np.where(np.isnan(tensor_data), None, tensor_data).tolist() if tensor_data is not None else None,
        'field_matrix_status': status_map,
        'pipeline_steps': pipeline_steps  # Zapisujemy listę kroków
    }

    with open(filepath, 'w') as f:
        json.dump(project_data, f, indent=4)

def load_project_json(filepath):
    """
    Wczytuje stan projektu z JSON.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Konwersja list z powrotem na numpy array
    if data.get('original_wavenumbers'):
        data['original_wavenumbers'] = np.array(data['original_wavenumbers'])
    
    if data.get('original_tensor_data'):
        # Zamiana None z powrotem na np.nan i konwersja na float
        # JSON zapisuje None jako null, pandas/numpy muszą to zamienić na nan
        arr = np.array(data['original_tensor_data'], dtype=float)
        # W numpy float None staje się nan automatycznie przy rzutowaniu, 
        # ale upewnijmy się, że typ jest poprawny
        data['original_tensor_data'] = arr
        
    return data

# --- modules/data_io.py (ZAKTUALIZOWANA FUNKCJA EKSPORTU) ---

def export_results_to_xlsx(filepath, wavenumbers, tensor_data, preprocessed_tensor, 
                           n_rows, m_cols, results_dict, pipeline_steps=None):
    """
    Eksportuje dane i wyniki wszystkich analiz do Excela.
    """
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        
        # --- 0. LOG PREPROCESSINGU ---
        if pipeline_steps:
            steps_data = []
            for i, step in enumerate(pipeline_steps):
                steps_data.append({
                    "Order": i + 1,
                    "Method": step['type'],
                    "Label": step['label'],
                    "Parameters": str(step['params'])
                })
            pd.DataFrame(steps_data).to_excel(writer, sheet_name='Preprocessing_Log', index=False)

        # --- 1. DANE PODSTAWOWE ---
        if wavenumbers is not None:
            pd.DataFrame(wavenumbers, columns=['Wavenumber']).to_excel(
                writer, sheet_name='Wavenumbers', index=False
            )

        if tensor_data is not None:
            data_2d = tensor_data.transpose(1, 2, 0).reshape((n_rows * m_cols), -1).T
            cols = [f"({r},{c})" for r in range(n_rows) for c in range(m_cols)]
            df = pd.DataFrame(data_2d, columns=cols, index=wavenumbers)
            df.index.name = "Wavenumber"
            df.to_excel(writer, sheet_name='Data_Raw')

        if preprocessed_tensor is not None:
            data_2d_proc = preprocessed_tensor.transpose(1, 2, 0).reshape((n_rows * m_cols), -1).T
            cols_proc = [f"({r},{c})" for r in range(n_rows) for c in range(m_cols)]
            df_proc = pd.DataFrame(data_2d_proc, columns=cols_proc, index=wavenumbers)
            df_proc.index.name = "Wavenumber"
            df_proc.to_excel(writer, sheet_name='Data_Preprocessed')

        # --- 2. PCA ---
        pca = results_dict.get('pca')
        if pca:
            pd.DataFrame(pca['scores'], 
                         columns=[f"PC{i+1}" for i in range(pca['scores'].shape[1])],
                         index=pca['labels']).to_excel(writer, sheet_name='PCA_Scores')
            pd.DataFrame(pca['loadings'], 
                         columns=[f"PC{i+1}" for i in range(pca['loadings'].shape[1])],
                         index=wavenumbers).to_excel(writer, sheet_name='PCA_Loadings', index_label="Wavenumber")
            pd.DataFrame(pca['variance'], columns=['Explained_Variance']).to_excel(writer, sheet_name='PCA_Variance')

        # --- 3. PARAFAC ---
        parafac = results_dict.get('parafac')
        if parafac:
            factors = parafac['factors']
            k = factors[0].shape[1]
            pd.DataFrame(factors[0], index=wavenumbers, 
                         columns=[f"Comp {i+1}" for i in range(k)]).to_excel(writer, sheet_name='PARAFAC_Spectra', index_label="Wavenumber")
            pd.DataFrame(factors[1], index=[f"Row {i}" for i in range(n_rows)],
                         columns=[f"Comp {i+1}" for i in range(k)]).to_excel(writer, sheet_name='PARAFAC_Trends_N')
            pd.DataFrame(factors[2], index=[f"Col {i}" for i in range(m_cols)],
                         columns=[f"Comp {i+1}" for i in range(k)]).to_excel(writer, sheet_name='PARAFAC_Trends_M')
            pd.DataFrame(parafac['weights'], columns=['Weight']).to_excel(writer, sheet_name='PARAFAC_Weights')

        # --- 4. TUCKER ---
        tucker = results_dict.get('tucker')
        if tucker:
            factors = tucker['factors']
            core = tucker['core']
            # factors[0] (Widma), factors[1] (N), factors[2] (M)
            pd.DataFrame(factors[0], index=wavenumbers).to_excel(writer, sheet_name='Tucker_Mode0_Spectra', index_label="Wavenumber")
            pd.DataFrame(factors[1]).to_excel(writer, sheet_name='Tucker_Mode1_Rows')
            pd.DataFrame(factors[2]).to_excel(writer, sheet_name='Tucker_Mode2_Cols')
            
            # Eksport plastrów rdzenia (Core)
            # Core jest 3D (r_w, r_n, r_m). Eksportujemy plastry wzdłuż r_m
            r_m = core.shape[2]
            for i in range(r_m):
                pd.DataFrame(core[:, :, i]).to_excel(writer, sheet_name=f'Tucker_Core_Slice_M{i}')

        # --- 5. MCR ---
        mcr = results_dict.get('mcr')
        if mcr:
            cols_c = [f"({r},{c})" for r in range(n_rows) for c in range(m_cols)]
            pd.DataFrame(mcr['C'], index=cols_c, 
                         columns=[f"Comp {i+1}" for i in range(mcr['rank'])]).to_excel(writer, sheet_name='MCR_Concentrations')
            pd.DataFrame(mcr['ST'], index=wavenumbers, 
                         columns=[f"Comp {i+1}" for i in range(mcr['rank'])]).to_excel(writer, sheet_name='MCR_Spectra', index_label="Wavenumber")

        # --- 6. SPEXFA ---
        spexfa = results_dict.get('spexfa')
        if spexfa:
            cols_c = spexfa['labels']
            pd.DataFrame(spexfa['C'], index=cols_c,
                         columns=[f"Comp {i+1}" for i in range(spexfa['rank'])]).to_excel(writer, sheet_name='SPEXFA_Concentrations')
            pd.DataFrame(spexfa['ST'], index=wavenumbers,
                         columns=[f"Comp {i+1}" for i in range(spexfa['rank'])]).to_excel(writer, sheet_name='SPEXFA_Spectra', index_label="Wavenumber")
        
        # --- 7. FA RANK (Malinowski) ---
        fa = results_dict.get('fa')
        if fa:
            metrics = pd.DataFrame({'Eigenvalues': fa['ev'], 'RE': fa['re'], 'IND': fa['ind']})
            metrics.to_excel(writer, sheet_name='FA_Metrics')
            # Abstrakcyjne widma (Loadings)
            pd.DataFrame(fa['vh'][:fa['max_k']].T, index=wavenumbers).to_excel(writer, sheet_name='FA_Abstract_Spectra', index_label="Wavenumber")

        # --- 8. PLS ---
        pls = results_dict.get('pls')
        if pls:
            pd.DataFrame(pls['coefs'], index=wavenumbers, columns=['Regression_Coefficients']).to_excel(writer, sheet_name='PLS_Coefs', index_label="Wavenumber")

        # --- 9. REKONSTRUKCJA ---
        recon = results_dict.get('recon')
        if recon:
            # Sprawdzamy czy to macierz czy tensor i spłaszczamy jeśli trzeba
            # Zapiszmy tylko rezydua i odtworzone dla zaznaczonych (jeśli mało) lub całość spłaszczoną
            # Dla uproszczenia: Zapisujemy spłaszczone rezydua (może być duży plik!)
            
            # Obsługa tensora (3D -> 2D)
            if 'residual_tensor' in recon:
                resid = recon['residual_tensor']
                rec_data = recon['recon_tensor']
                r_shp, n_shp, m_shp = resid.shape # w, n, m (dla PARAFAC) lub n, m, w?
                # Tensorly zwraca zazwyczaj tak jak input. U nas (W, N, M) w pamięci?
                # Nie, w main.py mamy (W, N, M) -> (wav, rows, cols)
                
                # Zapiszmy spłaszczone rezydua
                resid_2d = resid.reshape(resid.shape[0], -1) # (W, N*M)
                pd.DataFrame(resid_2d).to_excel(writer, sheet_name='Recon_Residuals')
            
            elif 'residuals' in recon: # PCA/FA (2D)
                pd.DataFrame(recon['residuals'].T, index=wavenumbers).to_excel(writer, sheet_name='Recon_Residuals', index_label="Wavenumber")

        # --- 10. MANIFOLD ---
        manifold = results_dict.get('manifold')
        if manifold:
             pd.DataFrame(manifold['scores'],
                          columns=["Comp 1", "Comp 2"],
                          index=manifold['labels']).to_excel(writer, sheet_name=f"{manifold['method_name']}_Scores")