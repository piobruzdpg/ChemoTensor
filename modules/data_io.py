# modules/data_io.py
import json
import pandas as pd
import numpy as np
import openpyxl


# --- KLASA POMOCNICZA DO ZAPISU JSON ---
class NumpyEncoder(json.JSONEncoder):
    """
    Automatycznie konwertuje typy NumPy (int64, float32, ndarray) 
    na zwykłe typy Pythona, żeby json.dump nie wyrzucał błędów.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- GŁÓWNE FUNKCJE ---

def load_single_csv(file_path):
    """Loads a single CSV file and returns wavenumbers and absorbance."""
    try:
        data = pd.read_csv(file_path, header=None, usecols=[0, 1]).values
        return data[:, 0], data[:, 1]
    except Exception as e:
        raise IOError(f"File read error: {e}")

def save_project_json(filepath, n_rows, m_cols, wavenumbers, tensor_data, status_map, pipeline_steps):
    """
    Saves project state to JSON, including preprocessing history.
    Uses NumpyEncoder to handle any potential numpy types in params.
    """
    project_data = {
        'version': '2.0',
        'n_rows': n_rows,
        'm_cols': m_cols,
        'original_wavenumbers': wavenumbers.tolist() if wavenumbers is not None else None,
        'original_tensor_data': np.where(np.isnan(tensor_data), None, tensor_data).tolist() if tensor_data is not None else None,
        'field_matrix_status': status_map,
        'pipeline_steps': pipeline_steps
    }

    with open(filepath, 'w') as f:
        # cls=NumpyEncoder załatwia sprawę typów numpy
        json.dump(project_data, f, indent=4, cls=NumpyEncoder)

def load_project_json(filepath):
    """
    Loads project state from JSON.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if data.get('original_wavenumbers'):
        data['original_wavenumbers'] = np.array(data['original_wavenumbers'])
    
    if data.get('original_tensor_data'):
        arr = np.array(data['original_tensor_data'], dtype=float)
        data['original_tensor_data'] = arr
        
    return data

def export_results_to_xlsx(filepath, wavenumbers, tensor_data, preprocessed_tensor, 
                           n_rows, m_cols, results_dict, pipeline_steps=None):
    """
    Exports data and analysis results to Excel.
    """
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        
        # --- 0. PREPROCESSING LOG ---
        if pipeline_steps:
            steps_data = []
            for i, step in enumerate(pipeline_steps):
                # Używamy .get() dla bezpieczeństwa - jeśli brakuje klucza, wstawi domyślną wartość
                steps_data.append({
                    "Order": i + 1,
                    "Method": step.get('type', 'Unknown'),
                    "Label": step.get('label', ''),
                    "Parameters": str(step.get('params', {}))
                })
            pd.DataFrame(steps_data).to_excel(writer, sheet_name='Preprocessing_Log', index=False)

        # --- 1. BASIC DATA ---
        if wavenumbers is not None:
            pd.DataFrame(wavenumbers, columns=['Wavenumber']).to_excel(
                writer, sheet_name='Wavenumbers', index=False
            )

        if tensor_data is not None:
            # Zakładamy tensor 3D -> spłaszczamy do 2D
            try:
                data_2d = tensor_data.transpose(1, 2, 0).reshape((n_rows * m_cols), -1).T
                cols = [f"({r},{c})" for r in range(n_rows) for c in range(m_cols)]
                df = pd.DataFrame(data_2d, columns=cols, index=wavenumbers)
                df.index.name = "Wavenumber"
                df.to_excel(writer, sheet_name='Data_Raw')
            except Exception as e:
                print(f"Warning: Could not export Data_Raw due to shape mismatch: {e}")

        if preprocessed_tensor is not None:
            try:
                # Tutaj też zabezpieczenie, gdyby preprocessing zmienił wymiary (np. unfold)
                if preprocessed_tensor.ndim == 3:
                    data_2d_proc = preprocessed_tensor.transpose(1, 2, 0).reshape((n_rows * m_cols), -1).T
                    cols_proc = [f"({r},{c})" for r in range(n_rows) for c in range(m_cols)]
                    df_proc = pd.DataFrame(data_2d_proc, columns=cols_proc, index=wavenumbers)
                    df_proc.index.name = "Wavenumber"
                    df_proc.to_excel(writer, sheet_name='Data_Preprocessed')
                else:
                    # Fallback dla danych 2D
                    pd.DataFrame(preprocessed_tensor).to_excel(writer, sheet_name='Data_Preprocessed_2D')
            except Exception as e:
                 print(f"Warning: Could not export Data_Preprocessed: {e}")

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
            pd.DataFrame(factors[0], index=wavenumbers).to_excel(writer, sheet_name='Tucker_Mode0_Spectra', index_label="Wavenumber")
            pd.DataFrame(factors[1]).to_excel(writer, sheet_name='Tucker_Mode1_Rows')
            pd.DataFrame(factors[2]).to_excel(writer, sheet_name='Tucker_Mode2_Cols')
            
            # Export core slices
            if core.ndim == 3:
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
            pd.DataFrame(fa['vh'][:fa['max_k']].T, index=wavenumbers).to_excel(writer, sheet_name='FA_Abstract_Spectra', index_label="Wavenumber")

        # --- 8. PLS ---
        pls = results_dict.get('pls')
        if pls:
            pd.DataFrame(pls['coefs'], index=wavenumbers, columns=['Regression_Coefficients']).to_excel(writer, sheet_name='PLS_Coefs', index_label="Wavenumber")

        # --- 9. RECONSTRUCTION ---
        recon = results_dict.get('recon')
        if recon:
            if 'residual_tensor' in recon:
                resid = recon['residual_tensor']
                # Zabezpieczenie przed błędnym reshape
                try:
                    resid_2d = resid.reshape(resid.shape[0], -1) # (W, N*M)
                    pd.DataFrame(resid_2d).to_excel(writer, sheet_name='Recon_Residuals')
                except Exception:
                     pd.DataFrame(resid[0,:,:]).to_excel(writer, sheet_name='Recon_Residuals_Slice0')
            
            elif 'residuals' in recon: # PCA/FA (2D)
                pd.DataFrame(recon['residuals'].T, index=wavenumbers).to_excel(writer, sheet_name='Recon_Residuals', index_label="Wavenumber")

        # --- 10. MANIFOLD ---
        manifold = results_dict.get('manifold')
        if manifold:
             pd.DataFrame(manifold['scores'],
                          columns=["Comp 1", "Comp 2"],
                          index=manifold['labels']).to_excel(writer, sheet_name=f"{manifold['method_name']}_Scores")