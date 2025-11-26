import tkinter
import json
from tkinter import filedialog, messagebox, Menu
import customtkinter as ctk
import matplotlib.figure
import matplotlib.backends.backend_tkagg
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, eye, spdiags
from scipy.sparse.linalg import spsolve
from scipy.linalg import pinv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  # Dla MSC
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE  # <-- NOWY IMPORT (v30)
import umap.umap_ as umap  # <-- NOWY IMPORT (v30, wymaga pip install umap-learn)
import tensorly as tl
from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNorm, ConstraintNonneg
import openpyxl

# --- Słownik motywów dla Matplotlib ---
THEME_COLORS = {
    'Light': {'bg_color': '#ffffff', 'text_color': '#000000', 'grid_color': '#cccccc', 'spine_color': '#bbbbbb',
              'label_bg': 'white'},
    'Dark': {'bg_color': '#2b2b2b', 'text_color': '#ffffff', 'grid_color': '#555555', 'spine_color': '#777777',
             'label_bg': '#2b2b2b'}
}

# --- Stałe Kolorów Statusu ---
STATUS_COLORS = {
    'EMPTY': ctk.ThemeManager.theme["CTkButton"]["fg_color"], 'MISSING': "gray50", 'LOADED': "#2ECC71",
    'ERROR': "#E74C3C", 'FILLED': "#0096FF"
}
SELECTED_BORDER_COLOR = "#3498DB"


# --- Główna klasa aplikacji ---
class ChemTensorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ChemTensor Explorer")
        self.geometry("1400x1000")

        # --- ZMIANA v33: Motyw na stałe ---
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

        self.n_var = ctk.StringVar(value="4")
        self.m_var = ctk.StringVar(value="4")

        self.field_matrix_widgets = {}
        self.field_matrix_status = {}
        self.selected_coords = set()

        self.original_tensor_data = None
        self.original_wavenumbers = None
        self.tensor_data = None
        self.wavenumbers = None

        # --- NOWE (v34): Lista plików i płaskie dane ---
        self.loaded_files = []  # List of dicts: {'path': str, 'name': str, 'data': np.array, 'active': bool}
        self.flat_tensor_data = None  # 2D array (W, K)

        self.range_min_var = ctk.StringVar()
        self.range_max_var = ctk.StringVar()

        # Preprocessing
        self.preprocessed_tensor = None
        self.sg_window_var = ctk.StringVar(value="5")
        self.sg_poly_var = ctk.StringVar(value="3")
        self.als_lambda_var = ctk.StringVar(value="1e6")
        self.als_p_var = ctk.StringVar(value="0.01")
        self.show_preprocessed_var = ctk.BooleanVar(value=False)
        self.pipeline_mode_var = ctk.BooleanVar(value=False)

        # Analiza
        self.current_plot_mode = 'SPECTRA'  # 'SPECTRA', 'PCA', 'RECONSTRUCTION', 'PARAFAC', 'MCR', 'TENSOR_RECONSTRUCTION', '3DCOS_SLICER', 'PLS_RESULTS', 'TUCKER_RESULTS', 'FA_RANK_RESULTS', 'FA_RECON_RESULTS', 'SPEXFA_RESULTS', 'HEATMAP', 'MANIFOLD_PLOT'

        self.fa_recon_components_var = ctk.StringVar(value="2")
        self.fa_results = None

        self.spexfa_n_components_var = ctk.StringVar(value="2")
        self.spexfa_results = None

        self.pca_n_components_var = ctk.StringVar(value="2")
        self.pca_recon_components_var = ctk.StringVar(value="2")
        self.pca_results = None

        self.parafac_rank_var = ctk.StringVar(value="2")
        self.parafac_non_negative_var = ctk.BooleanVar(value=False)
        self.parafac_results = None

        self.tucker_rank_w_var = ctk.StringVar(value="2")
        self.tucker_rank_n_var = ctk.StringVar(value="2")
        self.tucker_rank_m_var = ctk.StringVar(value="2")
        self.tucker_results = None

        self.mcr_n_components_var = ctk.StringVar(value="2")
        self.mcr_max_iter_var = ctk.StringVar(value="100")
        self.mcr_non_negative_var = ctk.BooleanVar(value=True)
        self.mcr_norm_var = ctk.BooleanVar(value=False)
        self.mcr_st_fix_var = ctk.StringVar(value="")
        self.mcr_st_init = None
        self.mcr_results = None

        self.tensor_recon_components_var = ctk.StringVar(value="2")
        self.tensor_recon_results = None

        self.cos_axis_var = ctk.StringVar(value="Analizuj Kolumny (M)")
        self.cos_slice_var = ctk.DoubleVar(value=0)
        self.cos_3d_results = None

        self.pls_target_var = ctk.StringVar(value="Wybierz Cel (y)...")
        self.pls_target_map = {}
        self.pls_results = None

        self.manifold_results = None  # NOWE (v30)

        self.zoom_rects = {}
        self.zoom_start = {}
        self.initial_lims = {}

        self.grid_columnconfigure(0, weight=2, minsize=450)
        self.grid_columnconfigure(1, weight=5)
        self.grid_rowconfigure(0, weight=1)

        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self._create_control_widgets()
        self._create_plot_canvas()
        self._create_field_matrix()
        self.update_plot()

    def _create_control_widgets(self):
        self.control_frame.grid_rowconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self.control_frame)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # --- Zakładka "Dane" ---
        data_tab = self.tabview.add("Dane")
        preprocess_tab = self.tabview.add("Preprocessing")
        analysis_tab = self.tabview.add("Analiza")
        
        data_scroll_frame = ctk.CTkScrollableFrame(data_tab, fg_color="transparent")
        data_scroll_frame.pack(fill="both", expand=True)
        data_scroll_frame.grid_columnconfigure(0, weight=1)

        preprocess_scroll_frame = ctk.CTkScrollableFrame(preprocess_tab, fg_color="transparent")
        preprocess_scroll_frame.pack(fill="both", expand=True)
        preprocess_scroll_frame.grid_columnconfigure(0, weight=1)

        analysis_scroll_frame = ctk.CTkScrollableFrame(analysis_tab, fg_color="transparent")
        analysis_scroll_frame.pack(fill="both", expand=True)
        analysis_scroll_frame.grid_columnconfigure(0, weight=1)

        # --- ZMIANA v35: Układ pionowy (Lista nad Siatką) ---
        data_scroll_frame.grid_columnconfigure(0, weight=1)
        # Usunięto podział na kolumny 0 i 1

        # --- SEKCJA 1: Lista Plików ---
        file_section_frame = ctk.CTkFrame(data_scroll_frame, fg_color="transparent")
        file_section_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        file_section_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(file_section_frame, text="Lista Plików Widmowych", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.file_list_frame = ctk.CTkScrollableFrame(file_section_frame, height=200) # Mniejsza wysokość, bo jest na górze
        self.file_list_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        file_actions_frame = ctk.CTkFrame(file_section_frame)
        file_actions_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        file_actions_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(file_actions_frame, text="Dodaj Pliki...", command=self._load_data_files).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(file_actions_frame, text="Zaznacz Wszystkie", command=lambda: self._set_all_files_active(True)).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(file_actions_frame, text="Odznacz Wszystkie", command=lambda: self._set_all_files_active(False)).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(file_actions_frame, text="Wyczyść Listę", command=self._clear_file_list, fg_color="#C0392B", hover_color="#E74C3C").grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        # --- SEKCJA 2: Konfiguracja Siatki ---
        config_frame = ctk.CTkFrame(data_scroll_frame)
        config_frame.grid(row=1, column=0, padx=5, pady=10, sticky="ew")
        config_frame.grid_columnconfigure((1, 3), weight=1)
        
        ctk.CTkLabel(config_frame, text="Konfiguracja Siatki (Tensor 3D)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=5, padx=5, pady=5, sticky="w")
        
        ctk.CTkLabel(config_frame, text="N (wiersze):").grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkEntry(config_frame, textvariable=self.n_var, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(config_frame, text="M (kolumny):").grid(row=1, column=2, padx=5, pady=5)
        ctk.CTkEntry(config_frame, textvariable=self.m_var, width=50).grid(row=1, column=3, padx=5, pady=5, sticky="w")
        ctk.CTkButton(config_frame, text="Utwórz Pustą", command=self.safe_create_field_matrix, width=80).grid(row=1, column=4, padx=5, pady=5)
        
        ctk.CTkButton(config_frame, text="Auto-rozmieść z Listy", command=self._auto_fill_grid_from_list, fg_color="#2980B9", hover_color="#3498DB").grid(row=2, column=0, columnspan=5, padx=5, pady=5, sticky="ew")

        # --- SEKCJA 3: Podgląd Siatki ---
        matrix_container = ctk.CTkFrame(data_scroll_frame, fg_color="transparent")
        matrix_container.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        matrix_container.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(matrix_container, text="Podgląd Siatki").grid(row=0, column=0, sticky="w", padx=5)
        self.field_matrix_frame = ctk.CTkFrame(matrix_container)
        self.field_matrix_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # --- SEKCJA 4: Zakres ---
        range_frame = ctk.CTkFrame(data_scroll_frame)
        range_frame.grid(row=3, column=0, padx=5, pady=10, sticky="ew")
        range_frame.grid_columnconfigure((1, 3), weight=1)
        ctk.CTkLabel(range_frame, text="Zakres Danych (Oś X)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=4, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(range_frame, text="Min:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(range_frame, textvariable=self.range_min_var, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(range_frame, text="Max:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(range_frame, textvariable=self.range_max_var, width=60).grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(range_frame, text="Zastosuj", command=self._apply_wavenumber_range).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(range_frame, text="Resetuj", command=self._reset_wavenumber_range).grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="ew")

        # --- SEKCJA 5: Projekt ---
        action_frame = ctk.CTkFrame(data_scroll_frame)
        action_frame.grid(row=4, column=0, padx=5, pady=10, sticky="ew")
        action_frame.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(action_frame, text="Zapisz Projekt...", command=self._save_project).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(action_frame, text="Wczytaj Projekt...", command=self._load_project).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(action_frame, text="Eksportuj Wyniki...", command=self._export_to_xlsx).grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # --- Zakładka "Wizualizacja" USUNIĘTA (zgodnie z prośbą) ---

        # --- Kontrolki Zakładki "Preprocessing" ---
        # --- Kontrolki Zakładki "Preprocessing" ---
        
        # Przełącznik "Pokaż dane po preprocessingu" (przeniesiony z Wizualizacji)
        self.show_preprocessed_switch = ctk.CTkCheckBox(preprocess_scroll_frame,
                                                        text="Pokaż dane po preprocessingu na wykresie",
                                                        variable=self.show_preprocessed_var,
                                                        command=self.update_plot)
        self.show_preprocessed_switch.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.pipeline_switch = ctk.CTkCheckBox(preprocess_scroll_frame,
                                               text="Kontynuuj przetwarzanie (zastosuj do wyniku)",
                                               variable=self.pipeline_mode_var)
        self.pipeline_switch.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        sg_frame = ctk.CTkFrame(preprocess_scroll_frame)
        sg_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        sg_frame.grid_columnconfigure(1, weight=1)
        sg_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(sg_frame, text="Wygładzanie / Pochodne", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0,
                                                                                                    columnspan=2,
                                                                                                    padx=10, pady=5,
                                                                                                    sticky="w")
        ctk.CTkLabel(sg_frame, text="Szerokość Okna:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(sg_frame, textvariable=self.sg_window_var).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(sg_frame, text="Stopień Wielomianu:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(sg_frame, textvariable=self.sg_poly_var).grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(sg_frame, text="Zastosuj Savitzky-Golay (2. Pochodna)", command=self._apply_sg_filter).grid(row=3,
                                                                                                                  column=0,
                                                                                                                  columnspan=2,
                                                                                                                  padx=10,
                                                                                                                  pady=10,
                                                                                                                  sticky="ew")

        norm_frame = ctk.CTkFrame(preprocess_scroll_frame)
        norm_frame.grid(row=3, column=0, padx=5, pady=10, sticky="ew")
        norm_frame.grid_columnconfigure((0, 1), weight=1)
        norm_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(norm_frame, text="Normalizacja", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0,
                                                                                            columnspan=2, padx=10,
                                                                                            pady=5, sticky="w")
        ctk.CTkButton(norm_frame, text="Zastosuj SNV", command=self._apply_snv).grid(row=1, column=0, padx=5, pady=5,
                                                                                     sticky="ew")
        ctk.CTkButton(norm_frame, text="Zastosuj Min-Max (0-1)", command=self._apply_min_max).grid(row=1, column=1,
                                                                                                   padx=5, pady=5,
                                                                                                   sticky="ew")
        ctk.CTkButton(norm_frame, text="Zastosuj Normę (Pole Powierzchni)", command=self._apply_l1_norm).grid(row=2,
                                                                                                              column=0,
                                                                                                              padx=5,
                                                                                                              pady=5,
                                                                                                              sticky="ew")
        ctk.CTkButton(norm_frame, text="Zastosuj MSC", command=self._apply_msc).grid(row=2, column=1, padx=5, pady=5,
                                                                                     sticky="ew")

        als_frame = ctk.CTkFrame(preprocess_scroll_frame)
        als_frame.grid(row=4, column=0, padx=5, pady=10, sticky="ew")
        als_frame.grid_columnconfigure(1, weight=1)
        als_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(als_frame, text="Korekcja Linii Bazowej (ALS)", font=ctk.CTkFont(weight="bold")).grid(row=0,
                                                                                                           column=0,
                                                                                                           columnspan=2,
                                                                                                           padx=10,
                                                                                                           pady=5,
                                                                                                           sticky="w")
        ctk.CTkLabel(als_frame, text="Lambda (gładkość):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(als_frame, textvariable=self.als_lambda_var).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(als_frame, text="P (asymetria):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(als_frame, textvariable=self.als_p_var).grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(als_frame, text="Zastosuj Korekcję ALS", command=self._apply_als).grid(row=3, column=0,
                                                                                             columnspan=2, padx=10,
                                                                                             pady=10, sticky="ew")

        # --- Kontrolki Zakładki "Analiza" ---
        fa_frame = ctk.CTkFrame(analysis_scroll_frame)
        fa_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        fa_frame.grid_columnconfigure(1, weight=1)
        fa_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(fa_frame, text="Analiza Faktorowa (Malinowski, pfa.m)", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        ctk.CTkButton(fa_frame, text="Uruchom Analizę Rangi (RE/IND)", command=self._run_fa_rank_analysis).grid(row=1,
                                                                                                                column=0,
                                                                                                                columnspan=2,
                                                                                                                padx=10,
                                                                                                                pady=10,
                                                                                                                sticky="ew")
        ctk.CTkLabel(fa_frame, text="L. faktorów (do rekonstrukcji):").grid(row=2, column=0, padx=10, pady=5,
                                                                            sticky="w")
        ctk.CTkEntry(fa_frame, textvariable=self.fa_recon_components_var).grid(row=2, column=1, padx=10, pady=5,
                                                                               sticky="ew")
        ctk.CTkButton(fa_frame, text="Uruchom Rekonstrukcję FA (pfa.m)", command=self._run_fa_reconstruction).grid(
            row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        spexfa_frame = ctk.CTkFrame(analysis_scroll_frame)
        spexfa_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        spexfa_frame.grid_columnconfigure(1, weight=1)
        spexfa_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(spexfa_frame, text="Izolacja Widm Faktorów (spexfa.m)", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(spexfa_frame, text="Liczba faktorów:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(spexfa_frame, textvariable=self.spexfa_n_components_var).grid(row=1, column=1, padx=10, pady=5,
                                                                                   sticky="ew")
        ctk.CTkButton(spexfa_frame, text="Uruchom Izolację Widm (spexfa)", command=self._run_spexfa).grid(row=2,
                                                                                                          column=0,
                                                                                                          columnspan=2,
                                                                                                          padx=10,
                                                                                                          pady=10,
                                                                                                          sticky="ew")

        pca_frame = ctk.CTkFrame(analysis_scroll_frame)
        pca_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        pca_frame.grid_columnconfigure(1, weight=1)
        pca_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(pca_frame, text="PCA (Analiza Głównych Składowych)", font=ctk.CTkFont(weight="bold")).grid(row=0,
                                                                                                                column=0,
                                                                                                                columnspan=2,
                                                                                                                padx=10,
                                                                                                                pady=5,
                                                                                                                sticky="w")
        ctk.CTkLabel(pca_frame, text="Liczba Komponentów:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(pca_frame, textvariable=self.pca_n_components_var).grid(row=1, column=1, padx=10, pady=5,
                                                                             sticky="ew")
        ctk.CTkButton(pca_frame, text="Uruchom PCA na Zaznaczonych Danych", command=self._run_pca).grid(row=2, column=0,
                                                                                                        columnspan=2,
                                                                                                        padx=10,
                                                                                                        pady=10,
                                                                                                        sticky="ew")

        recon_frame = ctk.CTkFrame(analysis_scroll_frame)
        recon_frame.grid(row=3, column=0, padx=5, pady=10, sticky="ew")
        recon_frame.grid_columnconfigure(1, weight=1)
        recon_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(recon_frame, text="Rekonstrukcja Danych PCA", font=ctk.CTkFont(weight="bold")).grid(row=0,
                                                                                                         column=0,
                                                                                                         columnspan=2,
                                                                                                         padx=10,
                                                                                                         pady=5,
                                                                                                         sticky="w")
        ctk.CTkLabel(recon_frame, text="Liczba składowych do odtworzenia:").grid(row=1, column=0, padx=10, pady=5,
                                                                                 sticky="w")
        ctk.CTkEntry(recon_frame, textvariable=self.pca_recon_components_var).grid(row=1, column=1, padx=10, pady=5,
                                                                                   sticky="ew")
        ctk.CTkButton(recon_frame, text="Odtwórz Dane (z PCA)", command=self._run_reconstruction).grid(row=2, column=0,
                                                                                                       columnspan=2,
                                                                                                       padx=10, pady=10,
                                                                                                       sticky="ew")

        parafac_frame = ctk.CTkFrame(analysis_scroll_frame)
        parafac_frame.grid(row=4, column=0, padx=5, pady=10, sticky="ew")
        parafac_frame.grid_columnconfigure(1, weight=1)
        parafac_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(parafac_frame, text="PARAFAC / Tensorly", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0,
                                                                                                     columnspan=2,
                                                                                                     padx=10, pady=5,
                                                                                                     sticky="w")
        ctk.CTkLabel(parafac_frame, text="Liczba Składowych (Ranga):").grid(row=1, column=0, padx=10, pady=5,
                                                                            sticky="w")
        ctk.CTkEntry(parafac_frame, textvariable=self.parafac_rank_var).grid(row=1, column=1, padx=10, pady=5,
                                                                             sticky="ew")
        ctk.CTkCheckBox(parafac_frame, text="Wymuś Nieujemność (NN-PARAFAC)",
                        variable=self.parafac_non_negative_var).grid(row=2, column=0, columnspan=2, padx=10, pady=10,
                                                                     sticky="w")
        ctk.CTkButton(parafac_frame, text="Uruchom PARAFAC na Całym Tensorze", command=self._run_parafac).grid(row=3,
                                                                                                               column=0,
                                                                                                               columnspan=2,
                                                                                                               padx=10,
                                                                                                               pady=10,
                                                                                                               sticky="ew")

        tucker_frame = ctk.CTkFrame(analysis_scroll_frame)
        tucker_frame.grid(row=5, column=0, padx=5, pady=10, sticky="ew")
        tucker_frame.grid_columnconfigure(1, weight=1)
        tucker_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(tucker_frame, text="Dekompozycja Tuckera (Tensorly)", font=ctk.CTkFont(weight="bold")).grid(row=0,
                                                                                                                 column=0,
                                                                                                                 columnspan=2,
                                                                                                                 padx=10,
                                                                                                                 pady=5,
                                                                                                                 sticky="w")
        ctk.CTkLabel(tucker_frame, text="Ranga Widm (W):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(tucker_frame, textvariable=self.tucker_rank_w_var).grid(row=1, column=1, padx=10, pady=5,
                                                                             sticky="ew")
        ctk.CTkLabel(tucker_frame, text="Ranga Wierszy (N):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(tucker_frame, textvariable=self.tucker_rank_n_var).grid(row=2, column=1, padx=10, pady=5,
                                                                             sticky="ew")
        ctk.CTkLabel(tucker_frame, text="Ranga Kolumn (M):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(tucker_frame, textvariable=self.tucker_rank_m_var).grid(row=3, column=1, padx=10, pady=5,
                                                                             sticky="ew")
        ctk.CTkButton(tucker_frame, text="Uruchom Analizę Tuckera", command=self._run_tucker).grid(row=4, column=0,
                                                                                                   columnspan=2,
                                                                                                   padx=10, pady=10,
                                                                                                   sticky="ew")

        mcr_frame = ctk.CTkFrame(analysis_scroll_frame)
        mcr_frame.grid(row=6, column=0, padx=5, pady=10, sticky="ew")
        mcr_frame.grid_columnconfigure(1, weight=1)
        mcr_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(mcr_frame, text="MCR-ALS (pyMCR)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0,
                                                                                              columnspan=2, padx=10,
                                                                                              pady=5, sticky="w")
        ctk.CTkLabel(mcr_frame, text="Liczba Składowych:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(mcr_frame, textvariable=self.mcr_n_components_var).grid(row=1, column=1, padx=10, pady=5,
                                                                             sticky="ew")
        ctk.CTkLabel(mcr_frame, text="Max Iteracji:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(mcr_frame, textvariable=self.mcr_max_iter_var).grid(row=2, column=1, padx=10, pady=5,
                                                                         sticky="ew")

        ctk.CTkCheckBox(mcr_frame, text="Wymuś Nieujemność (NNLS)", variable=self.mcr_non_negative_var).grid(row=3,
                                                                                                             column=0,
                                                                                                             columnspan=2,
                                                                                                             padx=10,
                                                                                                             pady=10,
                                                                                                             sticky="w")
        ctk.CTkCheckBox(mcr_frame, text="Wymuś sumę stężeń do 1 (Norm)", variable=self.mcr_norm_var).grid(row=4,
                                                                                                          column=0,
                                                                                                          columnspan=2,
                                                                                                          padx=10,
                                                                                                          pady=10,
                                                                                                          sticky="w")
        ctk.CTkButton(mcr_frame, text="Załaduj Znane Widmo (ST_fix)...", command=self._load_mcr_st_init).grid(row=5,
                                                                                                              column=0,
                                                                                                              columnspan=2,
                                                                                                              padx=10,
                                                                                                              pady=5,
                                                                                                              sticky="ew")
        ctk.CTkLabel(mcr_frame, text="Indeks(y) widm do zamrożenia (np. 0 lub 0,2):").grid(row=6, column=0, padx=10,
                                                                                           pady=5, sticky="w")
        ctk.CTkEntry(mcr_frame, textvariable=self.mcr_st_fix_var).grid(row=6, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(mcr_frame, text="Uruchom MCR-ALS na Całym Tensorze", command=self._run_mcr_als).grid(row=7,
                                                                                                           column=0,
                                                                                                           columnspan=2,
                                                                                                           padx=10,
                                                                                                           pady=10,
                                                                                                           sticky="ew")

        tensor_recon_frame = ctk.CTkFrame(analysis_scroll_frame)
        tensor_recon_frame.grid(row=7, column=0, padx=5, pady=10, sticky="ew")
        tensor_recon_frame.grid_columnconfigure((0, 1), weight=1)
        tensor_recon_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(tensor_recon_frame, text="Rekonstrukcja Tensora (PARAFAC/MCR)",
                     font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(tensor_recon_frame, text="Liczba składowych do odtworzenia:").grid(row=1, column=0, padx=10,
                                                                                        pady=5, sticky="w")
        ctk.CTkEntry(tensor_recon_frame, textvariable=self.tensor_recon_components_var).grid(row=1, column=1, padx=10,
                                                                                             pady=5, sticky="ew")
        ctk.CTkButton(tensor_recon_frame, text="Odtwórz z PARAFAC", command=self._run_parafac_recon).grid(row=2,
                                                                                                          column=0,
                                                                                                          padx=5,
                                                                                                          pady=10,
                                                                                                          sticky="ew")
        ctk.CTkButton(tensor_recon_frame, text="Odtwórz z MCR", command=self._run_mcr_recon).grid(row=2, column=1,
                                                                                                  padx=5, pady=10,
                                                                                                  sticky="ew")

        cos_frame = ctk.CTkFrame(analysis_scroll_frame)
        cos_frame.grid(row=8, column=0, padx=5, pady=10, sticky="ew")
        cos_frame.grid_columnconfigure(1, weight=1)
        cos_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(cos_frame, text="2D-COS (Noda-Ozaki)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0,
                                                                                                  columnspan=2, padx=10,
                                                                                                  pady=5, sticky="w")
        ctk.CTkLabel(cos_frame, text="Oś Perturbacji:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkOptionMenu(cos_frame, variable=self.cos_axis_var,
                          values=["Analizuj Wiersze (N)", "Analizuj Kolumny (M)"]).grid(row=1, column=1, padx=10,
                                                                                        pady=5, sticky="ew")
        ctk.CTkButton(cos_frame, text="Uruchom Analizę 3D-COS (Kostka)", command=self._run_3dcos).grid(row=2, column=0,
                                                                                                       columnspan=2,
                                                                                                       padx=10, pady=10,
                                                                                                       sticky="ew")

        self.cos_slider_label = ctk.CTkLabel(cos_frame, text="Plaster Modulatora (0):")
        self.cos_slider_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.cos_slider = ctk.CTkSlider(cos_frame, from_=0, to=1, number_of_steps=1, variable=self.cos_slice_var,
                                        command=self._on_cos_slider_change)
        self.cos_slider.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        self.cos_slider_label.grid_remove()
        self.cos_slider.grid_remove()

        pls_frame = ctk.CTkFrame(analysis_scroll_frame)
        pls_frame.grid(row=9, column=0, padx=5, pady=10, sticky="ew")
        pls_frame.grid_columnconfigure(1, weight=1)
        pls_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(pls_frame, text="Analiza PLS (Ważność Zmiennych)", font=ctk.CTkFont(weight="bold")).grid(row=0,
                                                                                                              column=0,
                                                                                                              columnspan=2,
                                                                                                              padx=10,
                                                                                                              pady=5,
                                                                                                              sticky="w")
        ctk.CTkLabel(pls_frame, text="Wybierz Cel (y):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.pls_target_menu = ctk.CTkOptionMenu(pls_frame, variable=self.pls_target_var,
                                                 values=["Brak wyników (uruchom MCR lub PARAFAC)"])
        self.pls_target_menu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(pls_frame, text="Uruchom Analizę PLS", command=self._run_pls).grid(row=2, column=0, columnspan=2,
                                                                                         padx=10, pady=10, sticky="ew")

        manifold_frame = ctk.CTkFrame(analysis_scroll_frame)
        manifold_frame.grid(row=10, column=0, padx=5, pady=10, sticky="ew")
        manifold_frame.grid_columnconfigure((0, 1), weight=1)
        manifold_frame.configure(border_width=1, border_color="gray50")
        ctk.CTkLabel(manifold_frame, text="Nieliniowa Redukcja Wymiaru", font=ctk.CTkFont(weight="bold")).grid(row=0,
                                                                                                               column=0,
                                                                                                               columnspan=2,
                                                                                                               padx=10,
                                                                                                               pady=5,
                                                                                                               sticky="w")
        ctk.CTkButton(manifold_frame, text="Uruchom UMAP (na zaznaczonych)", command=self._run_umap).grid(row=1,
                                                                                                          column=0,
                                                                                                          padx=5,
                                                                                                          pady=10,
                                                                                                          sticky="ew")
        ctk.CTkButton(manifold_frame, text="Uruchom t-SNE (na zaznaczonych)", command=self._run_tsne).grid(row=1,
                                                                                                           column=1,
                                                                                                           padx=5,
                                                                                                           pady=10,
                                                                                                           sticky="ew")

        # --- Zakładka Ustawienia została usunięta ---

    def safe_create_field_matrix(self):
        if self.original_tensor_data is not None:
            if not messagebox.askyesno("Potwierdzenie",
                                       "Utworzenie nowej siatki usunie wszystkie wczytane dane.\nCzy chcesz kontynuować?"):
                return
        self._create_field_matrix()
        self.focus_set()

    def _create_field_matrix(self):
        if not hasattr(self, 'field_matrix_frame'): return
        for widget in self.field_matrix_frame.winfo_children(): widget.destroy()
        self.field_matrix_widgets.clear()
        self.field_matrix_status.clear()
        self.selected_coords.clear()

        self.original_tensor_data = None
        self.original_wavenumbers = None
        self.tensor_data = None
        self.wavenumbers = None

        self.range_min_var.set("")
        self.range_max_var.set("")

        self._invalidate_preprocessing()
        self._invalidate_analysis()

        try:
            self.n_rows = int(self.n_var.get())
            self.m_cols = int(self.m_var.get())
            if self.n_rows <= 0 or self.m_cols <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Błąd", "N i M muszą być dodatnimi liczbami całkowitymi.")
            return

        for c_idx in range(self.m_cols): self.field_matrix_frame.grid_columnconfigure(c_idx, weight=1)
        for r_idx in range(self.n_rows):
            for c_idx in range(self.m_cols):
                coords = (r_idx, c_idx)
                cell_text = f"({r_idx}, {c_idx})\nPusty"
                btn = ctk.CTkButton(self.field_matrix_frame, text=cell_text, height=60, fg_color=STATUS_COLORS['EMPTY'],
                                    border_width=0)
                btn.grid(row=r_idx, column=c_idx, padx=1, pady=1, sticky="nsew")
                btn.bind("<Button-3>", lambda event, c=coords: self._on_cell_right_click(event, c))
                btn.bind("<Button-2>", lambda event, c=coords: self._on_cell_right_click(event, c))
                btn.configure(command=lambda c=coords: self._on_cell_left_click(c))
                self.field_matrix_widgets[coords] = btn
                self.field_matrix_status[coords] = 'EMPTY'

    def _on_cell_left_click(self, coords):
        if self.current_plot_mode not in ['TENSOR_RECONSTRUCTION', 'SPEXFA_RESULTS']:
            self.current_plot_mode = 'SPECTRA'

        status = self.field_matrix_status.get(coords)
        btn = self.field_matrix_widgets.get(coords)
        if not (status == 'LOADED' or status.startswith('FILLED')): return

        if coords in self.selected_coords:
            self.selected_coords.remove(coords)
            btn.configure(border_width=0)
        else:
            self.selected_coords.add(coords)
            btn.configure(border_width=3, border_color=SELECTED_BORDER_COLOR)
        self.update_plot()

    def _on_cell_right_click(self, event, coords):
        context_menu = Menu(self, tearoff=0)
        status = self.field_matrix_status.get(coords)
        if status != 'LOADED':
            context_menu.add_command(label="Oznacz jako BRAK DANYCH",
                                     command=lambda: self._mark_cell_as_missing(coords))
            context_menu.add_command(label="Oznacz jako Pusty", command=lambda: self._mark_cell_as_empty(coords))
        if status in ['MISSING', 'ERROR'] and self.original_tensor_data is not None:
            context_menu.add_separator()
            context_menu.add_command(label="Wypełnij zerami", command=lambda: self._fill_cell_with_zeros(coords))
            context_menu.add_command(label="Wypełnij średnią wiersza",
                                     command=lambda: self._fill_cell_with_row_mean(coords))
            context_menu.add_command(label="Wypełnij średnią kolumny",
                                     command=lambda: self._fill_cell_with_col_mean(coords))
        if status.startswith('LOADED') or status.startswith('FILLED'):
            context_menu.add_command(label="Resetuj (Oznacz jako Pusty)",
                                     command=lambda: self._mark_cell_as_empty(coords))
        context_menu.post(event.x_root, event.y_root)

    def _invalidate_preprocessing(self):
        if self.preprocessed_tensor is not None:
            self.preprocessed_tensor = None
            self.show_preprocessed_var.set(False)

    def _invalidate_analysis(self):
        self.pca_results = None
        self.parafac_results = None
        self.mcr_results = None
        self.tensor_recon_results = None
        self.cos_3d_results = None
        self.pls_results = None
        self.tucker_results = None
        self.fa_results = None
        self.spexfa_results = None
        self.manifold_results = None

        self.mcr_st_init = None
        self.mcr_st_fix_var.set("")

        if hasattr(self, 'pls_target_menu'):
            self.pls_target_map.clear()
            self.pls_target_menu.configure(values=["Brak wyników (uruchom MCR lub PARAFAC)"])
            self.pls_target_var.set("Brak wyników (uruchom MCR lub PARAFAC)")
        self.current_plot_mode = 'SPECTRA'

    def _update_cell_state(self, coords, status, text):
        self.field_matrix_status[coords] = status
        btn = self.field_matrix_widgets.get(coords)
        if btn:
            color_key = status.split('_')[0]
            color = STATUS_COLORS.get(color_key, STATUS_COLORS['EMPTY'])
            btn.configure(text=text, fg_color=color)
            if coords in self.selected_coords:
                self.selected_coords.remove(coords)
                btn.configure(border_width=0)
        self._invalidate_preprocessing()
        self._invalidate_analysis()

    def _apply_visuals_for_status(self, coords, status):
        btn = self.field_matrix_widgets.get(coords)
        if not btn: return
        text = f"{coords}\n{status}"
        color_key = status.split('_')[0]
        color = STATUS_COLORS.get(color_key, STATUS_COLORS['EMPTY'])
        btn.configure(text=text, fg_color=color, border_width=0)

    def _mark_cell_as_missing(self, coords):
        self._update_cell_state(coords, 'MISSING', f"{coords}\nBRAK")
        if self.original_tensor_data is not None:
            self.original_tensor_data[:, coords[0], coords[1]] = np.nan
        if self.tensor_data is not None:
            self.tensor_data[:, coords[0], coords[1]] = np.nan

    def _mark_cell_as_empty(self, coords):
        self._update_cell_state(coords, 'EMPTY', f"{coords}\nPusty")
        if self.original_tensor_data is not None:
            self.original_tensor_data[:, coords[0], coords[1]] = np.nan
        if self.tensor_data is not None:
            self.tensor_data[:, coords[0], coords[1]] = np.nan

    def _fill_cell_with_zeros(self, coords):
        if self.original_tensor_data is None: return
        self.original_tensor_data[:, coords[0], coords[1]] = 0.0
        self.tensor_data[:, coords[0], coords[1]] = 0.0
        self._update_cell_state(coords, 'FILLED_ZERO', f"{coords}\nWypełniono (0)")
        self.update_plot()

    def _fill_cell_with_row_mean(self, coords):
        if self.original_tensor_data is None: return
        try:
            mean_spectrum = np.nanmean(self.original_tensor_data[:, coords[0], :], axis=1)
            if np.all(np.isnan(mean_spectrum)): raise ValueError("Brak danych w wierszu")
            self.original_tensor_data[:, coords[0], coords[1]] = mean_spectrum
            self.tensor_data[:, coords[0], coords[1]] = mean_spectrum
            self._update_cell_state(coords, 'FILLED_ROW_MEAN', f"{coords}\nWypełniono (Śr. W.)")
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można obliczyć średniej dla wiersza {coords[0]}.\n{e}")

    def _fill_cell_with_col_mean(self, coords):
        if self.original_tensor_data is None: return
        try:
            mean_spectrum = np.nanmean(self.original_tensor_data[:, :, coords[1]], axis=1)
            if np.all(np.isnan(mean_spectrum)): raise ValueError("Brak danych w kolumnie")
            self.original_tensor_data[:, coords[0], coords[1]] = mean_spectrum
            self.tensor_data[:, coords[0], coords[1]] = mean_spectrum
            self._update_cell_state(coords, 'FILLED_COL_MEAN', f"{coords}\nWypełniono (Śr. K.)")
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można obliczyć średniej dla kolumny {coords[1]}.\n{e}")

    def _load_data_files(self):
        file_paths = filedialog.askopenfilenames(title=f"Wybierz pliki CSV/DPT...",
                                                 filetypes=[("Pliki danych", "*.csv *.dpt"), ("Pliki CSV", "*.csv"), ("Pliki DPT", "*.dpt"), ("Wszystkie pliki", "*.*")])
        if not file_paths:
            return

        is_first_file = (self.original_wavenumbers is None) and (len(self.loaded_files) == 0)
        
        for file_path in file_paths:
            file_name = file_path.split('/')[-1]
            try:
                data = pd.read_csv(file_path, header=None, usecols=[0, 1]).values
                current_wavenumbers, current_absorbance = data[:, 0], data[:, 1]

                if is_first_file:
                    self.original_wavenumbers = current_wavenumbers
                    is_first_file = False
                
                if self.original_wavenumbers is not None and not np.array_equal(self.original_wavenumbers, current_wavenumbers):
                     messagebox.showwarning("Ostrzeżenie", f"Plik {file_name} ma inną oś X (wavenumbers) i został pominięty.")
                     continue

                self.loaded_files.append({
                    'path': file_path,
                    'name': file_name,
                    'data': current_absorbance,
                    'active': True
                })
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać pliku {file_name}:\n{e}")
        
        self._refresh_file_list_ui()
        self._update_flat_data()
        
        if self.wavenumbers is None and self.original_wavenumbers is not None:
             self._reset_wavenumber_range()
             
        self.update_plot() # Update plot immediately
        self.focus_set()

    def _refresh_file_list_ui(self):
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        for idx, file_info in enumerate(self.loaded_files):
            row_frame = ctk.CTkFrame(self.file_list_frame)
            row_frame.pack(fill="x", pady=2)
            row_frame.grid_columnconfigure(1, weight=1)

            # Checkbox (Active)
            chk_var = ctk.BooleanVar(value=file_info['active'])
            chk = ctk.CTkCheckBox(row_frame, text="", variable=chk_var, width=24, command=lambda i=idx, v=chk_var: self._toggle_file_active(i, v))
            chk.grid(row=0, column=0, padx=5)

            # Name
            lbl = ctk.CTkLabel(row_frame, text=f"{idx+1}. {file_info['name']}", anchor="w")
            lbl.grid(row=0, column=1, sticky="ew", padx=5)

            # Controls
            btn_up = ctk.CTkButton(row_frame, text="▲", width=24, command=lambda i=idx: self._move_file_up(i))
            btn_up.grid(row=0, column=2, padx=2)
            
            btn_down = ctk.CTkButton(row_frame, text="▼", width=24, command=lambda i=idx: self._move_file_down(i))
            btn_down.grid(row=0, column=3, padx=2)
            
            btn_del = ctk.CTkButton(row_frame, text="X", width=24, fg_color="#E74C3C", hover_color="#C0392B", command=lambda i=idx: self._remove_file(i))
            btn_del.grid(row=0, column=4, padx=2)

    def _toggle_file_active(self, idx, var):
        self.loaded_files[idx]['active'] = var.get()
        self._update_flat_data()
        self.update_plot() # Update plot on toggle
        self.focus_set()

    def _move_file_up(self, idx):
        if idx > 0:
            self.loaded_files[idx], self.loaded_files[idx-1] = self.loaded_files[idx-1], self.loaded_files[idx]
            self._refresh_file_list_ui()
            self._update_flat_data()
        self.focus_set()

    def _move_file_down(self, idx):
        if idx < len(self.loaded_files) - 1:
            self.loaded_files[idx], self.loaded_files[idx+1] = self.loaded_files[idx+1], self.loaded_files[idx]
            self._refresh_file_list_ui()
            self._update_flat_data()
        self.focus_set()

    def _remove_file(self, idx):
        del self.loaded_files[idx]
        if len(self.loaded_files) == 0:
            self.original_wavenumbers = None
            self.flat_tensor_data = None
        self._refresh_file_list_ui()
        self._update_flat_data()
        self.update_plot()
        self.focus_set()

    def _set_all_files_active(self, active):
        for f in self.loaded_files:
            f['active'] = active
        self._refresh_file_list_ui()
        self._update_flat_data()
        self.update_plot()
        self.focus_set()

    def _clear_file_list(self):
        if not self.loaded_files: return
        if messagebox.askyesno("Potwierdzenie", "Czy na pewno chcesz usunąć wszystkie pliki z listy?"):
            self.loaded_files.clear()
            self.original_wavenumbers = None
            self.flat_tensor_data = None
            self._refresh_file_list_ui()
            self._update_flat_data()
            self.update_plot()
        self.focus_set()

    def _update_flat_data(self):
        # Updates self.flat_tensor_data based on active files
        active_files = [f for f in self.loaded_files if f['active']]
        if not active_files:
            self.flat_tensor_data = None
            return
        
        # Stack data: (Wavenumbers, Files)
        try:
            self.flat_tensor_data = np.column_stack([f['data'] for f in active_files])
        except Exception as e:
            print(f"Error updating flat data: {e}")
            self.flat_tensor_data = None

    def _auto_fill_grid_from_list(self):
        active_files = [f for f in self.loaded_files if f['active']]
        if not active_files:
            messagebox.showinfo("Info", "Brak aktywnych plików na liście.")
            return
        
        try:
            n = int(self.n_var.get())
            m = int(self.m_var.get())
        except ValueError:
            messagebox.showerror("Błąd", "Nieprawidłowe wymiary N lub M.")
            return

        if len(active_files) > n * m:
            if not messagebox.askyesno("Ostrzeżenie", f"Masz {len(active_files)} plików, a siatka mieści tylko {n*m}. Nadmiarowe pliki zostaną pominięte. Kontynuować?"):
                return
        
        # Preserve wavenumbers because _create_field_matrix clears them
        saved_wavenumbers = self.original_wavenumbers
        
        self.safe_create_field_matrix() # Reset grid
        
        # Restore wavenumbers
        self.original_wavenumbers = saved_wavenumbers
        
        # Fill grid
        for idx, file_info in enumerate(active_files):
            if idx >= n * m: break
            r, c = divmod(idx, m)
            coords = (r, c)
            
            # Update grid logic (similar to old _load_data_files but from memory)
            if self.original_tensor_data is None:
                 if self.original_wavenumbers is None:
                      # Should not happen if saved_wavenumbers was valid, but safety check
                      messagebox.showerror("Błąd", "Brak danych osi X (wavenumbers).")
                      return
                 self.original_tensor_data = np.full((len(self.original_wavenumbers), n, m), np.nan)
            
            self.original_tensor_data[:, r, c] = file_info['data']
            self.field_matrix_status[coords] = 'LOADED'
            self._apply_visuals_for_status(coords, 'LOADED')
            self.field_matrix_widgets[coords].configure(text=f"{coords}\n{file_info['name']}")
        
        # Update tensor_data copy
        self.tensor_data = self.original_tensor_data.copy() if self.original_tensor_data is not None else None
        messagebox.showinfo("Sukces", "Siatka została wypełniona plikami z listy.")
        self._reset_wavenumber_range()
        self.update_plot()
        self.focus_set()

    def _select_all_active(self):
        """Zaznacza wszystkie komórki, które mają status LOADED lub FILLED."""
        print("Zaznaczanie wszystkich aktywnych komórek...")
        for coords, status in self.field_matrix_status.items():
            if status.startswith('LOADED') or status.startswith('FILLED'):
                if coords not in self.selected_coords:
                    self.selected_coords.add(coords)
                    btn = self.field_matrix_widgets.get(coords)
                    if btn:
                        btn.configure(border_width=3, border_color=SELECTED_BORDER_COLOR)
        self.update_plot()

    def _deselect_all(self):
        """Odznacza wszystkie zaznaczone komórki."""
        if not self.selected_coords:
            return  # Nic do zrobienia

        print("Odznaczanie wszystkich komórek...")
        # Musimy zrobić kopię, ponieważ będziemy modyfikować zbiór
        coords_to_clear = list(self.selected_coords)
        self.selected_coords.clear()

        for coords in coords_to_clear:
            btn = self.field_matrix_widgets.get(coords)
            if btn:
                btn.configure(border_width=0)
        self.update_plot()

    def _apply_wavenumber_range(self):
        if self.original_tensor_data is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane.")
            return
        try:
            min_val = float(self.range_min_var.get())
            max_val = float(self.range_max_var.get())
        except ValueError:
            messagebox.showerror("Błąd", "Wartości Min i Max muszą być liczbami.")
            return
        if min_val >= max_val:
            messagebox.showerror("Błąd", "Min musi być mniejszy niż Max.")
            return

        mask = np.where((self.original_wavenumbers >= min_val) & (self.original_wavenumbers <= max_val))[0]

        if len(mask) == 0:
            messagebox.showerror("Błąd", "Brak punktów danych w podanym zakresie.")
            return

        self.wavenumbers = self.original_wavenumbers[mask]
        self.tensor_data = self.original_tensor_data[mask, :, :]

        print(f"Zakres ograniczony do {len(self.wavenumbers)} punktów ({min_val} - {max_val}).")

        self._invalidate_preprocessing()
        self._invalidate_analysis()
        self.update_plot()
        self.focus_set()
        self.focus_set()

    def _reset_wavenumber_range(self):
        if self.original_wavenumbers is None:
            return

        self.wavenumbers = np.copy(self.original_wavenumbers)
        if self.original_tensor_data is not None:
            self.tensor_data = np.copy(self.original_tensor_data)

        if self.wavenumbers is not None:
            self.range_min_var.set(f"{np.min(self.wavenumbers):.2f}")
            self.range_max_var.set(f"{np.max(self.wavenumbers):.2f}")
        else:
            self.range_min_var.set("")
            self.range_max_var.set("")

        print("Zakres zresetowany do pełnego zakresu.")

        self._invalidate_preprocessing()
        self._invalidate_analysis()
        self.update_plot()
        self.focus_set()
        self.focus_set()

    def _get_data_source(self):
        if self.pipeline_mode_var.get() and self.preprocessed_tensor is not None:
            print("Pobieranie danych: Źródło = Dane Przetworzone")
            return self.preprocessed_tensor
        elif self.tensor_data is not None:
            print("Pobieranie danych: Źródło = Dane Robocze (Siatka)")
            return self.tensor_data
        elif self.flat_tensor_data is not None:
            print("Pobieranie danych: Źródło = Dane Robocze (Lista)")
            # Wrap flat data into a mock 3D structure (1, K, W) or handle directly
            # Currently preprocessing expects 3D (1, N, M) or (W, N, M) depending on usage
            # But our tensor_data is (W, N, M).
            # flat_tensor_data is (W, K).
            # We can reshape it to (W, K, 1) to reuse the same logic
            w, k = self.flat_tensor_data.shape
            return self.flat_tensor_data.reshape(w, k, 1)
        else:
            messagebox.showerror("Błąd", "Brak danych. Wczytaj najpierw pliki.")
            return None

    def _process_data(self, processing_function, name="Przetwarzanie"):
        data_source = self._get_data_source()
        if data_source is None:
            return

        self._invalidate_analysis()

        try:
            self.preprocessed_tensor = np.copy(data_source)
            processing_function(self.preprocessed_tensor)

            self.show_preprocessed_var.set(True)
            self.update_plot()
            messagebox.showinfo("Sukces", f"Operacja '{name}' została pomyślnie zastosowana.")
        except Exception as e:
            messagebox.showerror(f"Błąd {name}", f"Wystąpił błąd:\n{e}")
            self._invalidate_preprocessing()

    def _als_baseline(self, y, lam, p, n_iter=10):
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

    def _apply_als(self):
        try:
            lam = float(self.als_lambda_var.get())
            p = float(self.als_p_var.get())
            if not (0 < p < 1): raise ValueError("P (asymetria) musi być między 0 a 1.")
            if lam <= 0: raise ValueError("Lambda (gładkość) musi być dodatnia.")
        except ValueError as e:
            messagebox.showerror("Błąd Walidacji", f"Błędne parametry ALS: {e}")
            return

        def als_logic(tensor):
            print(f"Stosowanie korekcji ALS (lambda={lam}, p={p})...")
            # tensor shape: (W, N, M) or (W, K, 1)
            w, n, m = tensor.shape
            for r in range(n):
                for c in range(m):
                    if not np.isnan(tensor[0, r, c]):
                        spectrum = tensor[:, r, c]
                        baseline = self._als_baseline(spectrum, lam, p)
                        tensor[:, r, c] = spectrum - baseline

        self._process_data(als_logic, name="Korekcja ALS")
        self.focus_set()

    def _apply_sg_filter(self):
        try:
            window = int(self.sg_window_var.get())
            poly = int(self.sg_poly_var.get())
            if window % 2 == 0 or window <= 0: raise ValueError("Szerokość okna musi być nieparzysta i dodatnia")
            if poly >= window: raise ValueError("Stopień wielomianu musi być mniejszy niż okno")
        except ValueError as e:
            messagebox.showerror("Błąd Walidacji", f"{e}")
            return

        def sg_logic(tensor):
            print(f"Stosowanie filtru SG (okno={window}, wielomian={poly}, deriv=2)...")
            w, n, m = tensor.shape
            for r in range(n):
                for c in range(m):
                    if not np.isnan(tensor[0, r, c]):
                        tensor[:, r, c] = savgol_filter(tensor[:, r, c], window, poly, deriv=2)

        self._process_data(sg_logic, name="Savitzky-Golay")
        self.focus_set()

    def _apply_snv(self):
        def snv_logic(tensor):
            print("Stosowanie SNV...")
            w, n, m = tensor.shape
            for r in range(n):
                for c in range(m):
                    if not np.isnan(tensor[0, r, c]):
                        spectrum = tensor[:, r, c]
                        mean = np.mean(spectrum)
                        std = np.std(spectrum)
                        if std > 1e-8:
                            tensor[:, r, c] = (spectrum - mean) / std
                        else:
                            tensor[:, r, c] = spectrum - mean

        self._process_data(snv_logic, name="SNV")
        self.focus_set()

    def _apply_min_max(self):
        def min_max_logic(tensor):
            print("Stosowanie Skalowania Min-Max...")
            w, n, m = tensor.shape
            for r in range(n):
                for c in range(m):
                    if not np.isnan(tensor[0, r, c]):
                        spectrum = tensor[:, r, c]
                        min_val = np.min(spectrum)
                        range_val = np.max(spectrum) - min_val
                        if range_val > 1e-8:
                            tensor[:, r, c] = (spectrum - min_val) / range_val
                        else:
                            tensor[:, r, c] = 0.5

        self._process_data(min_max_logic, name="Min-Max (0-1)")
        self.focus_set()

    def _apply_l1_norm(self):
        def l1_logic(tensor):
            print("Stosowanie Normy L1 (Pole Powierzchni)...")
            w, n, m = tensor.shape
            for r in range(n):
                for c in range(m):
                    if not np.isnan(tensor[0, r, c]):
                        spectrum = tensor[:, r, c]
                        area = np.sum(np.abs(spectrum))
                        if area > 1e-8:
                            tensor[:, r, c] = spectrum / area

        self._process_data(l1_logic, name="Norma (Pole Powierzchni)")
        self.focus_set()

    def _apply_msc(self):
        def msc_logic(tensor):
            print("Stosowanie MSC...")
            w, n, m = tensor.shape
            all_spectra = []
            for r in range(n):
                for c in range(m):
                    if not np.isnan(tensor[0, r, c]):
                        all_spectra.append(tensor[:, r, c])
            if not all_spectra:
                raise ValueError("Brak danych do obliczenia widma referencyjnego.")

            mean_spectrum = np.mean(np.array(all_spectra), axis=0).reshape(-1, 1)
            model = LinearRegression()

            for r in range(n):
                for c in range(m):
                    if not np.isnan(tensor[0, r, c]):
                        spectrum = tensor[:, r, c].reshape(-1, 1)
                        model.fit(mean_spectrum, spectrum)
                        intercept = model.intercept_[0]
                        slope = model.coef_[0][0]

                        if np.abs(slope) > 1e-8:
                            tensor[:, r, c] = (tensor[:, r, c] - intercept) / slope
                        else:
                            tensor[:, r, c] = tensor[:, r, c] - intercept

        self._process_data(msc_logic, name="MSC")
        self.focus_set()

    def _get_data_matrix_from_selection(self):
        # 1. Try Grid Selection first
        if self.selected_coords and self.tensor_data is not None:
            data_source = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
            if data_source is None: return None, None, None

            data_matrix = []
            sample_labels = []
            sorted_coords_list = sorted(list(self.selected_coords))

            r_coords = sorted(list(set([c[0] for c in sorted_coords_list])))
            c_coords = sorted(list(set([c[1] for c in sorted_coords_list])))
            n_rows_sub = len(r_coords)
            m_cols_sub = len(c_coords)
            rc_sub_to_index_map = {}

            for i, coords in enumerate(sorted_coords_list):
                r, c = coords
                spectrum = data_source[:, r, c]
                if not np.isnan(spectrum).any():
                    data_matrix.append(spectrum)
                    sample_labels.append(str(coords))
                    r_idx_sub = r_coords.index(r)
                    c_idx_sub = c_coords.index(c)
                    rc_sub_to_index_map[(r_idx_sub, c_idx_sub)] = i

            if len(data_matrix) < 2:
                messagebox.showerror("Błąd", "Zbyt mało ważnych próbek (min. 2).")
                return None, None, None

            map_info = {
                "type": "grid",
                "n_rows_sub": n_rows_sub,
                "m_cols_sub": m_cols_sub,
                "rc_sub_to_index_map": rc_sub_to_index_map
            }
            return np.array(data_matrix), sample_labels, map_info

        # 2. Fallback to List Selection (Active Files)
        active_files = [f for f in self.loaded_files if f['active']]
        if active_files:
            # Check if we should use preprocessed data
            use_preprocessed = self.show_preprocessed_var.get() and self.preprocessed_tensor is not None
            
            if use_preprocessed:
                 # preprocessed_tensor is (W, K, 1) -> we need (K, W)
                 # Check if dimensions match active files count
                 w, k, _ = self.preprocessed_tensor.shape
                 if k == len(active_files):
                      data_matrix = self.preprocessed_tensor.reshape(w, k).T
                 else:
                      # Fallback if counts mismatch (e.g. changed selection after processing)
                      # Ideally we should re-process or warn, but for now fallback to raw
                      if self.flat_tensor_data is None: self._update_flat_data()
                      data_matrix = self.flat_tensor_data.T if self.flat_tensor_data is not None else None
            else:
                # Check if we have flat data
                if self.flat_tensor_data is None:
                     self._update_flat_data()
                data_matrix = self.flat_tensor_data.T if self.flat_tensor_data is not None else None
            
            if data_matrix is None:
                messagebox.showerror("Błąd", "Brak danych na liście.")
                return None, None, None

            sample_labels = [f['name'] for f in active_files]
            
            if len(data_matrix) < 2:
                 messagebox.showerror("Błąd", "Zbyt mało aktywnych plików (min. 2).")
                 return None, None, None

            map_info = {"type": "list"}
            return data_matrix, sample_labels, map_info

        messagebox.showerror("Błąd", "Brak zaznaczonych danych (na siatce lub liście).")
        return None, None, None

    def _run_pca(self):
        X, labels, _ = self._get_data_matrix_from_selection()
        if X is None: return

        try:
            n_components = int(self.pca_n_components_var.get())
            if n_components < 2: n_components = 2
            if n_components > X.shape[0]: n_components = X.shape[0]
        except ValueError:
            messagebox.showerror("Błąd", "Liczba komponentów musi być liczbą całkowitą.")
            return
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.T).T

            pca = PCA(n_components=n_components)
            scores = pca.fit_transform(X_scaled)
            self.pca_results = {
                'scores': scores, 'loadings': pca.components_.T,
                'variance': pca.explained_variance_ratio_, 'labels': labels,
                'pca_model': pca, 'scaler': scaler, 'X_original': X
            }
            self.current_plot_mode = 'PCA'
            self.update_plot()
            messagebox.showinfo("Sukces", "Analiza PCA zakończona.")
            self.focus_set()
        except Exception as e:
            messagebox.showerror("Błąd PCA", f"Wystąpił błąd podczas analizy PCA:\n{e}")
            self.pca_results = None

    def _run_reconstruction(self):
        if self.pca_results is None:
            messagebox.showerror("Błąd", "Najpierw uruchom analizę PCA.")
            return
        try:
            k = int(self.pca_recon_components_var.get())
            max_k = self.pca_results['scores'].shape[1]
            if not (0 < k <= max_k):
                messagebox.showerror("Błąd Walidacji", f"Liczba składowych musi być liczbą od 1 do {max_k}.")
                return
        except ValueError:
            messagebox.showerror("Błąd", "Liczba składowych musi być liczbą całkowitą.")
            return
        try:
            scores = self.pca_results['scores'];
            pca = self.pca_results['pca_model']
            scaler = self.pca_results['scaler'];
            X_original = self.pca_results['X_original']
            scores_k = np.copy(scores);
            scores_k[:, k:] = 0
            X_recon_scaled = pca.inverse_transform(scores_k)
            X_reconstructed = scaler.inverse_transform(X_recon_scaled.T).T
            residuals = X_original - X_reconstructed
            self.pca_results['recon_k'] = k
            self.pca_results['X_reconstructed'] = X_reconstructed
            self.pca_results['residuals'] = residuals
            self.current_plot_mode = 'RECONSTRUCTION'
            self.update_plot()
            messagebox.showinfo("Sukces", f"Dane odtworzone przy użyciu {k} składowych.")
            self.focus_set()
        except Exception as e:
            messagebox.showerror("Błąd Rekonstrukcji", f"Wystąpił błąd:\n{e}")

    def _update_pls_target_options(self):
        self.pls_target_map.clear()
        options = []

        if self.mcr_results:
            k = self.mcr_results['rank']
            for i in range(k):
                name = f"MCR Stężenie Skł. {i + 1}"
                self.pls_target_map[name] = self.mcr_results['C'][:, i]
                options.append(name)

        if self.parafac_results:
            k = self.parafac_results['factors'][0].shape[1]
            factors = self.parafac_results['factors']
            for i in range(k):
                name_n = f"PARAFAC Trend N Skł. {i + 1}"
                trend_n = np.repeat(factors[1][:, i], self.m_cols)
                self.pls_target_map[name_n] = trend_n
                options.append(name_n)

                name_m = f"PARAFAC Trend M Skł. {i + 1}"
                trend_m = np.tile(factors[2][:, i], self.n_rows)
                self.pls_target_map[name_m] = trend_m
                options.append(name_m)

        if not options:
            options = ["Brak wyników (uruchom MCR lub PARAFAC)"]

        self.pls_target_menu.configure(values=options)
        self.pls_target_var.set(options[0])

    def _run_parafac(self):
        if self.tensor_data is None:
            messagebox.showerror("Błąd", "Brak danych w siatce (Tensorze). PARAFAC wymaga danych 3D.\nUżyj 'Auto-rozmieść z Listy' w zakładce Dane.")
            return
        data_source = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
        if data_source is None:
            messagebox.showerror("Błąd", "Brak danych do analizy PARAFAC.")
            return
        try:
            rank = int(self.parafac_rank_var.get())
            if rank <= 0: raise ValueError("Ranga musi być dodatnia")
        except ValueError as e:
            messagebox.showerror("Błąd Walidacji", f"Niepoprawna liczba składowych: {e}")
            return
        try:
            tensor_no_nan = np.nan_to_num(data_source, nan=0.0)
            non_negative = self.parafac_non_negative_var.get()

            if non_negative:
                print(f"Uruchamianie NN-PARAFAC z rangą {rank}...")
                weights, factors = tl.decomposition.non_negative_parafac(tensor_no_nan, rank=rank, init='random',
                                                                         n_iter_max=100)
                msg = "Analiza NN-PARAFAC zakończona."
            else:
                print(f"Uruchamianie PARAFAC z rangą {rank}...")
                weights, factors = tl.decomposition.parafac(tensor_no_nan, rank=rank, n_iter_max=100)
                msg = "Analiza PARAFAC zakończona."

            self.parafac_results = {'weights': weights, 'factors': factors}
            self.current_plot_mode = 'PARAFAC'
            self._update_pls_target_options()
            self.update_plot()
            messagebox.showinfo("Sukces", msg)
            self.focus_set()
        except Exception as e:
            messagebox.showerror("Błąd PARAFAC", f"Wystąpił błąd podczas analizy PARAFAC:\n{e}")
            self.parafac_results = None

    def _load_mcr_st_init(self):
        if self.wavenumbers is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane (i ustaw zakres), aby zdefiniować oś liczb falowych.")
            return

        file_paths = filedialog.askopenfilenames(
            title="Wybierz plik(i) ze znanymi widmami...",
            filetypes=[("Pliki CSV", "*.csv"), ("Wszystkie pliki", "*.*")]
        )
        if not file_paths:
            return

        loaded_spectra = []
        for file_path in file_paths:
            try:
                data = pd.read_csv(file_path, header=None, usecols=[0, 1]).values
                current_wavenumbers, current_absorbance = data[:, 0], data[:, 1]

                if not np.array_equal(self.wavenumbers, current_wavenumbers):
                    messagebox.showerror("Błąd Osi X", f"Oś liczb falowych w pliku {file_path.split('/')[-1]} "
                                                       f"nie jest identyczna z aktywnym zakresem danych w aplikacji.")
                    self.mcr_st_init = None
                    return

                loaded_spectra.append(current_absorbance)

            except Exception as e:
                messagebox.showerror("Błąd Odczytu Pliku",
                                     f"Nie udało się wczytać pliku {file_path.split('/')[-1]}.\n{e}")
                self.mcr_st_init = None
                return

        self.mcr_st_init = np.array(loaded_spectra)

        suggested_indices = ",".join(map(str, range(len(file_paths))))
        self.mcr_st_fix_var.set(suggested_indices)

        messagebox.showinfo("Sukces", f"Pomyślnie wczytano {len(file_paths)} znanych widm.\n"
                                      f"Wprowadź indeksy do zamrożenia (np. {suggested_indices})")
        self.focus_set()

    def _run_mcr_als(self):
        # Determine Data Source
        D_matrix = None
        is_grid_data = False
        
        # Try Grid
        data_source = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
        if data_source is not None:
             tensor_no_nan = np.nan_to_num(data_source, nan=0.0)
             w, n, m = tensor_no_nan.shape
             D_matrix = tensor_no_nan.transpose(1, 2, 0).reshape((n * m), w)
             is_grid_data = True
        
        # Try List if Grid is missing
        elif self.flat_tensor_data is not None:
             D_matrix = self.flat_tensor_data.T # (K, W)
             is_grid_data = False
        
        if D_matrix is None:
            messagebox.showerror("Błąd", "Brak danych do analizy MCR-ALS.")
            return

        try:
            rank = int(self.mcr_n_components_var.get())
            if rank <= 0: raise ValueError("Liczba składowych musi być dodatnia")
            
            max_iter = int(self.mcr_max_iter_var.get())
            if max_iter <= 0: max_iter = 100
        except ValueError as e:
            messagebox.showerror("Błąd Walidacji", f"Niepoprawne parametry: {e}")
            return
        try:
            n_samples, n_features = D_matrix.shape

            c_constraints = []
            st_constraints = []

            if self.mcr_non_negative_var.get():
                print("Uruchamianie MCR-ALS z regresorem NNLS (stężenia i widma nieujemne)...")
                c_regr = NNLS()
                st_regr = NNLS()

                if self.mcr_norm_var.get():
                    print("Dodawanie ograniczenia: Suma stężeń = 1")
                    c_constraints.append(ConstraintNorm())
            else:
                print("Uruchamianie MCR-ALS z regresorem OLS (stężenia i widma mogą być ujemne)...")
                c_regr = OLS()
                st_regr = OLS()

            st_init = None
            st_fix_indices = []

            st_fix_str = self.mcr_st_fix_var.get()
            if st_fix_str:
                try:
                    st_fix_indices = [int(i.strip()) for i in st_fix_str.split(',')]
                except ValueError:
                    messagebox.showerror("Błąd Walidacji",
                                         "Indeksy do zamrożenia muszą być listą liczb oddzielonych przecinkami (np. 0 lub 0,2).")
                    return

            if self.mcr_st_init is not None:
                if len(st_fix_indices) != self.mcr_st_init.shape[0]:
                    messagebox.showerror("Błąd Walidacji",
                                         f"Wczytałeś {self.mcr_st_init.shape[0]} widm, ale podałeś {len(st_fix_indices)} indeksów do zamrożenia.\nLiczba ta musi być identyczna.")
                    return

                print(f"Używanie {len(st_fix_indices)} znanych widm jako ST_init...")
                st_init = np.random.rand(rank, n_features)

                for i, fix_idx in enumerate(st_fix_indices):
                    if fix_idx >= rank:
                        messagebox.showerror("Błąd Walidacji",
                                             f"Indeks {fix_idx} jest poza zakresem. Ranga to {rank}, więc maks. indeks to {rank - 1}.")
                        return
                    st_init[fix_idx, :] = self.mcr_st_init[i, :]

            else:
                if st_fix_indices:
                    messagebox.showerror("Błąd Walidacji",
                                         "Podałeś indeksy do zamrożenia, ale nie wczytałeś żadnych znanych widm (użyj przycisku 'Załaduj Znane Widmo').")
                    return
                st_init = np.random.rand(rank, n_features)

            mcr = McrAR(max_iter=max_iter,
                        c_regr=c_regr,
                        st_regr=st_regr,
                        c_constraints=c_constraints,
                        st_constraints=st_constraints)

            mcr.fit(D_matrix, ST=st_init, st_fix=st_fix_indices)

            self.mcr_results = {
                'C': mcr.C_opt_,  # (n_samples, k)
                'ST': mcr.ST_opt_.T,  # (w, k)
                'rank': rank,
                'is_grid_data': is_grid_data
            }

            self.current_plot_mode = 'MCR'
            self._update_pls_target_options()
            self.update_plot()
            messagebox.showinfo("Sukces", "Analiza MCR-ALS zakończona.")
            self.focus_set()

        except ImportError:
            messagebox.showerror("Błąd Importu",
                                 "Nie znaleziono biblioteki 'pymcr'.\nUpewnij się, że jest zainstalowana: pip install pymcr")
        except Exception as e:
            messagebox.showerror("Błąd MCR-ALS", f"Wystąpił błąd podczas analizy MCR-ALS:\n{e}")
            self.mcr_results = None

    def _get_source_tensor_for_recon(self):
        data_source = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
        if data_source is None:
            messagebox.showerror("Błąd", "Brak danych źródłowych.")
            return None
        return np.nan_to_num(data_source, nan=0.0)

    def _run_parafac_recon(self):
        if self.parafac_results is None:
            messagebox.showerror("Błąd", "Najpierw uruchom analizę PARAFAC.")
            return
        try:
            k = int(self.tensor_recon_components_var.get())
            max_k = self.parafac_results['factors'][0].shape[1]
            if not (0 < k <= max_k):
                messagebox.showerror("Błąd Walidacji", f"Liczba składowych musi być liczbą od 1 do {max_k}.")
                return
        except ValueError:
            messagebox.showerror("Błąd", "Liczba składowych musi być liczbą całkowitą.")
            return
        try:
            print(f"Rekonstrukcja z {k} składowych PARAFAC...")
            weights = self.parafac_results['weights'][:k]
            factors = [f[:, :k] for f in self.parafac_results['factors']]
            reconstructed_tensor = tl.cp_to_tensor((weights, factors))
            source_tensor = self._get_source_tensor_for_recon()
            if source_tensor is None: return
            residual_tensor = source_tensor - reconstructed_tensor

            self.tensor_recon_results = {
                'source': 'parafac', 'k': k,
                'recon_tensor': reconstructed_tensor,
                'residual_tensor': residual_tensor,
                'source_tensor': source_tensor,
                'components': factors[0]
            }
            self.current_plot_mode = 'TENSOR_RECONSTRUCTION'
            self.update_plot()
            messagebox.showinfo("Sukces", f"Tensor odtworzony z {k} składowych PARAFAC.")
            self.focus_set()
        except Exception as e:
            messagebox.showerror("Błąd Rekonstrukcji PARAFAC", f"Wystąpił błąd:\n{e}")

    def _run_mcr_recon(self):
        if self.mcr_results is None:
            messagebox.showerror("Błąd", "Najpierw uruchom analizę MCR-ALS.")
            return
        if not self.mcr_results['is_grid_data']:
            messagebox.showerror("Błąd", "Rekonstrukcja tensora MCR jest dostępna tylko dla danych z siatki (3D).")
            return
        try:
            k = int(self.tensor_recon_components_var.get())
            max_k = self.mcr_results['rank']
            if not (0 < k <= max_k):
                messagebox.showerror("Błąd Walidacji", f"Liczba składowych musi być liczbą od 1 do {max_k}.")
                return
        except ValueError:
            messagebox.showerror("Błąd", "Liczba składowych musi być liczbą całkowitą.")
            return
        try:
            print(f"Rekonstrukcja z {k} składowych MCR...")
            C_k = self.mcr_results['C'][:, :k]  # (n*m, k)
            ST_k = self.mcr_results['ST'][:, :k]  # (w, k)
            D_recon_k = C_k @ ST_k.T  # (n*m, w)

            w = ST_k.shape[0]
            reconstructed_tensor = D_recon_k.reshape((self.n_rows, self.m_cols, w)).transpose(2, 0, 1)  # (w, n, m)
            source_tensor = self._get_source_tensor_for_recon()
            if source_tensor is None: return

            residual_tensor = source_tensor - reconstructed_tensor

            self.tensor_recon_results = {
                'source': 'mcr', 'k': k,
                'recon_tensor': reconstructed_tensor,
                'residual_tensor': residual_tensor,
                'source_tensor': source_tensor,
                'components': ST_k
            }
            self.current_plot_mode = 'TENSOR_RECONSTRUCTION'
            self.update_plot()
            messagebox.showinfo("Sukces", f"Tensor odtworzony z {k} składowych MCR.")
            self.focus_set()
        except Exception as e:
            messagebox.showerror("Błąd Rekonstrukcji MCR", f"Wystąpił błąd:\n{e}")

    def _calculate_2dcos(self, D):
        n_features, n_samples = D.shape
        if n_samples < 2:
            raise ValueError("2D-COS wymaga co najmniej 2 próbek (punktów perturbacji).")

        mean_spec = D.mean(axis=1, keepdims=True)
        Y = D - mean_spec

        Phi = (1.0 / (n_samples - 1)) * (Y @ Y.T)

        N = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    N[i, j] = 1.0 / (j - i)

        Psi = (1.0 / (n_samples - 1)) * (Y @ N @ Y.T)

        return Phi, Psi

    def _run_3dcos(self):
        # Use _get_data_source to support both grid and list data
        data_source = self._get_data_source()
        if data_source is None:
            # Error already shown in _get_data_source if None
            return

        try:
            tensor_no_nan = np.nan_to_num(data_source, nan=0.0)
            w, n, m = tensor_no_nan.shape
            axis_choice = self.cos_axis_var.get()

            phi_list = []
            psi_list = []

            if axis_choice == "Analizuj Wiersze (N)":
                print(f"Uruchamianie 3D-COS (Modulator: Wiersze N={n}, Perturbacja: Kolumny M={m})...")
                modulator_size = n
                if m < 2: 
                     # If we have list data reshaped as (W, K, 1), M=1.
                     # Row analysis requires M >= 2 (perturbation axis).
                     messagebox.showerror("Błąd", "Analiza wierszy wymaga co najmniej 2 kolumn (M >= 2).")
                     return

                for i in range(n):
                    slice_2d = tensor_no_nan[:, i, :]
                    phi, psi = self._calculate_2dcos(slice_2d)
                    phi_list.append(phi)
                    psi_list.append(psi)

            else:  # "Analizuj Kolumny (M)"
                print(f"Uruchamianie 3D-COS (Modulator: Kolumny M={m}, Perturbacja: Wiersze N={n})...")
                modulator_size = m
                if n < 2: 
                    messagebox.showerror("Błąd", "Analiza kolumn wymaga co najmniej 2 wierszy (N >= 2).")
                    return

                for j in range(m):
                    slice_2d = tensor_no_nan[:, :, j]
                    phi, psi = self._calculate_2dcos(slice_2d)
                    phi_list.append(phi)
                    psi_list.append(psi)

            phi_cube = np.stack(phi_list, axis=-1)
            psi_cube = np.stack(psi_list, axis=-1)

            self.cos_3d_results = {
                'phi': phi_cube,
                'psi': psi_cube
            }

            # Fix ZeroDivisionError if modulator_size is 1
            steps = modulator_size - 1
            if steps < 1: steps = 1
            
            self.cos_slider.configure(to=modulator_size - 1, number_of_steps=steps)
            self.cos_slice_var.set(0)

            self.current_plot_mode = '3DCOS_SLICER'
            self.update_plot()
            messagebox.showinfo("Sukces", "Analiza 3D-COS zakończona.\nUżyj suwaka, aby przeglądać plastry.")
            self.focus_set()

        except Exception as e:
            messagebox.showerror("Błąd 2D-COS", f"Wystąpił błąd podczas analizy 2D-COS:\n{e}")
            self._invalidate_analysis()

    def _on_cos_slider_change(self, value):
        self.cos_slice_var.set(int(float(value)))
        self.update_plot()

    def _visualize_3dcos_cube(self):
        pass

    def _run_pls(self):
        target_name = self.pls_target_var.get()
        if target_name not in self.pls_target_map:
            messagebox.showerror("Błąd",
                                 "Wybierz prawidłowy cel (y) z listy.\nUruchom analizę MCR lub PARAFAC, aby wypełnić listę.")
            return
        y = self.pls_target_map[target_name]

        data_source = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
        if data_source is None:
            messagebox.showerror("Błąd", "Brak danych (X) do analizy.")
            return

        try:
            tensor_no_nan = np.nan_to_num(data_source, nan=0.0)
            w, n, m = tensor_no_nan.shape
            X = tensor_no_nan.transpose(1, 2, 0).reshape((n * m), w)  # (próbki, cechy)

            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Niezgodność wymiarów X ({X.shape[0]}) i y ({y.shape[0]})")

            print(f"Uruchamianie PLS (n_comp=1) dla celu: {target_name}")
            pls = PLSRegression(n_components=1)
            pls.fit(X, y)

            self.pls_results = {
                'coefs': pls.coef_.flatten(),  # (w,)
                'target_name': target_name
            }

            self.current_plot_mode = 'PLS_RESULTS'
            self.update_plot()
            messagebox.showinfo("Sukces", "Analiza PLS zakończona.")
            self.focus_set()

        except Exception as e:
            messagebox.showerror("Błąd PLS", f"Wystąpił błąd podczas analizy PLS:\n{e}")
            self.pls_results = None

    def _run_tucker(self):
        if self.tensor_data is None:
            messagebox.showerror("Błąd", "Brak danych w siatce (Tensorze). Tucker wymaga danych 3D.\nUżyj 'Auto-rozmieść z Listy' w zakładce Dane.")
            return
        data_source = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
        if data_source is None:
            messagebox.showerror("Błąd", "Brak danych do analizy Tuckera.")
            return

        try:
            rank_w = int(self.tucker_rank_w_var.get())
            rank_n = int(self.tucker_rank_n_var.get())
            rank_m = int(self.tucker_rank_m_var.get())
            if not (rank_w > 0 and rank_n > 0 and rank_m > 0):
                raise ValueError("Rangi muszą być dodatnimi liczbami całkowitymi")
        except ValueError as e:
            messagebox.showerror("Błąd Walidacji", f"Niepoprawne rangi: {e}")
            return

        try:
            tensor_no_nan = np.nan_to_num(data_source, nan=0.0)

            print(f"Uruchamianie Dekompozycji Tuckera z rangami ({rank_w}, {rank_n}, {rank_m})...")

            core, factors = tl.decomposition.tucker(tensor_no_nan, rank=[rank_w, rank_n, rank_m])

            self.tucker_results = {
                'core': core,
                'factors': factors
            }

            self.current_plot_mode = 'TUCKER_RESULTS'
            self.update_plot()
            messagebox.showinfo("Sukces", "Analiza Tuckera zakończona.")
            self.focus_set()

        except Exception as e:
            messagebox.showerror("Błąd Tuckera", f"Wystąpił błąd podczas analizy Tuckera:\n{e}")
            self.tucker_results = None

    def _run_fa_rank_analysis(self):
        """Tłumaczy logikę 'pfa.m' do obliczania EV, RE i IND na ZAZNACZONYCH danych."""
        X, labels, map_info = self._get_data_matrix_from_selection()
        if X is None: return

        try:
            r, c = X.shape  # r = próbki, c = cechy

            malinowski_r = c
            malinowski_c = r

            sm = min(malinowski_r, malinowski_c)
            lg = max(malinowski_r, malinowski_c)

            max_k = sm - 1
            if max_k < 1:
                messagebox.showerror("Błąd", f"Niewystarczająca liczba próbek lub cech do analizy FA (min. 2x2).")
                return

            print("Uruchamianie SVD dla Analizy Faktorowej...")
            u, s_vec, vh = np.linalg.svd(X, full_matrices=False)  # X = U @ S @ Vh

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

            print("Analiza Rangi (RE/IND) zakończona.")

            self.fa_results = {
                'u': u, 's_vec': s_vec, 'vh': vh,
                'ev': ev[:max_k], 're': re, 'ind': ind, 'max_k': max_k,
                'X_original': X,
                'labels': labels,
                'map_info': map_info
            }

            self.current_plot_mode = 'FA_RANK_RESULTS'
            self.update_plot()
            self.focus_set()

        except Exception as e:
            messagebox.showerror("Błąd Analizy Faktorowej", f"Wystąpił błąd podczas analizy RE/IND:\n{e}")
            self.fa_results = None

    def _run_fa_reconstruction(self):
        if self.fa_results is None:
            messagebox.showerror("Błąd", "Najpierw uruchom 'Analizę Rangi (RE/IND)', aby wykonać SVD.")
            return

        try:
            k = int(self.fa_recon_components_var.get())
            max_k = len(self.fa_results['s_vec'])
            if not (0 < k <= max_k):
                messagebox.showerror("Błąd Walidacji", f"Liczba faktorów musi być między 1 a {max_k}.")
                return
        except ValueError:
            messagebox.showerror("Błąd", "Liczba faktorów musi być liczbą całkowitą.")
            return

        try:
            print(f"Rekonstrukcja FA z {k} faktorami...")
            u = self.fa_results['u']
            s_vec = self.fa_results['s_vec']
            vh = self.fa_results['vh']
            X_original = self.fa_results['X_original']  # (próbki, cechy)

            u_k = u[:, :k]
            s_k = np.diag(s_vec[:k])
            vh_k = vh[:k, :]

            X_recon = u_k @ s_k @ vh_k  # (próbki, cechy)

            residuals = X_original - X_recon

            loadings = vh_k.T  # (cechy, k)
            scores = u_k @ s_k  # (próbki, k)

            self.fa_results['recon_k'] = k
            self.fa_results['recon_X_original'] = X_original
            self.fa_results['recon_X_recon'] = X_recon
            self.fa_results['recon_residuals'] = residuals
            self.fa_results['recon_loadings_k'] = loadings
            self.fa_results['recon_scores_k'] = scores

            self.current_plot_mode = 'FA_RECON_RESULTS'
            self.update_plot()
            messagebox.showinfo("Sukces", "Rekonstrukcja FA zakończona.")
            self.focus_set()

        except Exception as e:
            messagebox.showerror("Błąd Rekonstrukcji FA", f"Wystąpił błąd:\n{e}")

    def _run_spexfa(self):
        """Tłumaczy logikę 'spexfa.m' na ZAZNACZONYCH danych."""
        X_selected, labels, map_info = self._get_data_matrix_from_selection()
        if X_selected is None: return

        try:
            n = int(self.spexfa_n_components_var.get())  # Liczba faktorów
        except ValueError:
            messagebox.showerror("Błąd", "Liczba faktorów musi być liczbą całkowitą.")
            return

        try:
            D = X_selected.T  # (cechy, próbki) - tak jak w spexfa.m
            r, c = D.shape  # r = cechy, c = próbki
            sm = min(r, c)
            lg = max(r, c)

            if not (0 < n < sm):
                messagebox.showerror("Błąd Walidacji", f"Liczba faktorów (n) musi być dodatnia i mniejsza niż {sm}.")
                return

            print("Uruchamianie SVD dla SPEXFA...")
            u, s_vec, vh = np.linalg.svd(D, full_matrices=False)
            s_mat = np.diag(s_vec)

            u_n = u[:, :n]
            s_n = s_mat[:n, :n]
            vh_n = vh[:n, :]
            dr = u_n @ s_n @ vh_n  # (cechy, próbki)

            ev = s_vec ** 2
            sev = np.sum(ev[n:])
            re = np.sqrt(sev / (r * (c - n)))
            cutoff = 5 * re * np.sqrt(n)

            ubar = u_n @ s_n  # (cechy, n)

            print("Wyszukiwanie kluczowych zmiennych (key set)...")
            ubar_norm = np.linalg.norm(ubar, axis=1)
            mask = ubar_norm < cutoff
            ubar_masked = np.copy(ubar)
            ubar_masked[mask, :] = np.nan

            ubar_norm_clean_indices = ~mask
            if np.sum(ubar_norm_clean_indices) == 0:
                messagebox.showerror("Błąd SPEXFA",
                                     "Brak 'czystych' zmiennych. Wszystkie cechy poniżej progu szumu.\nSpróbuj zmniejszyć liczbę faktorów (n).")
                return

            ubar_norm_clean = np.linalg.norm(ubar_masked[ubar_norm_clean_indices, :], axis=1, keepdims=True)
            ubar_masked[ubar_norm_clean_indices, :] = (ubar_masked[ubar_norm_clean_indices, :] * np.sqrt(
                c)) / ubar_norm_clean

            key = [np.nanargmin(np.abs(ubar_masked[:, 0]))]

            for k in range(1, n):
                w = ubar_masked[:, :k + 1]
                ky = np.zeros((k + 1, k + 1))
                for j in range(k):
                    ky[j, :] = w[key[j], :]

                dt = np.zeros(r)
                for i in range(r):
                    if np.isnan(ubar_masked[i, 0]):
                        dt[i] = -np.inf
                    else:
                        ky[k, :] = w[i, :]
                        dt[i] = np.abs(np.linalg.det(ky))

                key.append(np.argmax(dt))

            w_all = u_n @ s_n
            iter_count = 0
            max_iters = 20
            while iter_count < max_iters:
                tkey = key.copy()
                for j in range(n):
                    tset = np.zeros((n, n))
                    for i in range(n):
                        tset[i, :] = w_all[key[i], :]

                    dt = np.zeros(r)
                    for k_row in range(r):
                        if np.isnan(ubar_masked[k_row, 0]):
                            dt[k_row] = -np.inf
                        else:
                            tset[j, :] = w_all[k_row, :]
                            dt[k_row] = np.abs(np.linalg.det(tset))

                    key[j] = np.argmax(dt)

                if np.array_equal(key, tkey):
                    break
                iter_count += 1

            print(f"SPEXFA: Zestaw kluczy znaleziony w {iter_count} iteracjach: {key}")

            conc_transposed = dr[key, :]
            spex = dr @ pinv(conc_transposed)

            self.spexfa_results = {
                'C': conc_transposed.T,  # (próbki, faktory)
                'ST': spex,  # (cechy, faktory)
                'rank': n,
                'labels': labels,
                'map_info': map_info
            }

            self.current_plot_mode = 'SPEXFA_RESULTS'
            self.update_plot()
            messagebox.showinfo("Sukces", "Izolacja Widm Faktorów (spexfa) zakończona.")
            self.focus_set()

        except Exception as e:
            messagebox.showerror("Błąd SPEXFA", f"Wystąpił błąd podczas analizy SPEXFA:\n{e}")
            self.spexfa_results = None

    def _run_manifold(self, method_name):
        """Wspólna funkcja dla UMAP i t-SNE."""
        X, labels, _ = self._get_data_matrix_from_selection()
        if X is None: return

        try:
            print(f"Uruchamianie {method_name} na {X.shape[0]} próbkach...")
            if method_name == 'UMAP':
                model = umap.UMAP(n_components=2, random_state=42)
            else:  # t-SNE
                perplexity = min(30.0, X.shape[0] - 1.0)
                model = TSNE(n_components=2, random_state=42, perplexity=perplexity)

            embedding = model.fit_transform(X)

            self.manifold_results = {
                'scores': embedding,
                'labels': labels,
                'method_name': method_name
            }

            self.current_plot_mode = 'MANIFOLD_PLOT'
            self.update_plot()
            messagebox.showinfo("Sukces", f"Analiza {method_name} zakończona.")
            self.focus_set()

        except ImportError:
            messagebox.showerror("Błąd Importu",
                                 "Nie znaleziono biblioteki 'umap-learn'.\nUpewnij się, że jest zainstalowana: pip install umap-learn")
        except Exception as e:
            messagebox.showerror(f"Błąd {method_name}", f"Wystąpił błąd:\n{e}")
            self.manifold_results = None

    def _run_umap(self):
        self._run_manifold('UMAP')

    def _run_tsne(self):
        self._run_manifold('t-SNE')

    def _run_heatmap(self):
        """Przełącza tryb wykresu na HEATMAP."""
        self.current_plot_mode = 'HEATMAP'
        self.update_plot()

    def _create_plot_canvas(self):
        self.fig = matplotlib.figure.Figure(figsize=(10, 7), dpi=100)
        # self.axes will be created in update_plot
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.toolbar = matplotlib.backends.backend_tkagg.NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        self.canvas.mpl_connect('button_press_event', self._on_canvas_button_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self._on_canvas_button_release)

    def _change_theme(self, new_theme_str):
        pass  # Usunięte

    def _apply_theme_to_axis(self, ax, theme):
        ax.set_facecolor(theme['bg_color'])
        ax.xaxis.label.set_color(theme['text_color'])
        ax.yaxis.label.set_color(theme['text_color'])
        ax.title.set_color(theme['text_color'])
        ax.tick_params(axis='x', colors=theme['text_color'])
        ax.tick_params(axis='y', colors=theme['text_color'])
        for spine in ax.spines.values():
            spine.set_color(theme['spine_color'])

    def _apply_theme_to_legend(self, legend, theme):
        if legend:
            legend.get_frame().set_facecolor(theme['bg_color'])
            legend.get_frame().set_edgecolor(theme['spine_color'])
            for text in legend.get_texts():
                text.set_color(theme['text_color'])

    def _set_wavenumber_axis_inverted(self, ax):
        if self.wavenumbers is not None and len(self.wavenumbers) > 0:
            ax.set_xlim(np.max(self.wavenumbers), np.min(self.wavenumbers))

    def update_plot(self):
        if not hasattr(self, 'canvas'): return

        # Determine layout based on mode
        single_plot_modes = ['SPECTRA', 'HEATMAP', 'MANIFOLD_PLOT', 'PLS_RESULTS']
        
        self.fig.clear()
        
        if self.current_plot_mode in single_plot_modes:
             ax = self.fig.add_subplot(111)
             self.axes = np.array([[ax]]) # Mock 2D array structure for compatibility
        else:
             self.axes = self.fig.subplots(2, 2)

        theme = THEME_COLORS[ctk.get_appearance_mode()]
        self.fig.patch.set_facecolor(theme['bg_color'])
        
        # Apply theme to all axes
        for ax in self.fig.get_axes():
             self._apply_theme_to_axis(ax, theme)

        if hasattr(self, 'cos_slider_label'):
            self.cos_slider_label.grid_remove()
            self.cos_slider.grid_remove()

        if self.current_plot_mode == 'SPECTRA':
            ax = self.fig.get_axes()[0]
            
            title_suffix = ""
            use_preprocessed = self.show_preprocessed_var.get()
            data_source = self.preprocessed_tensor if use_preprocessed and self.preprocessed_tensor is not None else self.tensor_data

            if use_preprocessed:
                if self.preprocessed_tensor is None:
                    y_label = "Absorbancja"
                    title_suffix = " (Brak danych po preprocessingu!)"
                else:
                    y_label = "Intensywność (Przetworzone)"
                    title_suffix = " (Po Preprocessingu)"
            else:
                y_label = "Absorbancja"
                title_suffix = " (Surowe Dane)"

            if self.wavenumbers is not None:
                ax.set_xlabel(f"Liczba falowa (wymiar: {len(self.wavenumbers)})")
                self._set_wavenumber_axis_inverted(ax)
            else:
                ax.set_xlabel("Liczba falowa")
            ax.set_ylabel(y_label)

            if not self.selected_coords:
                # Try plotting active files from list if no grid selection
                active_files = [f for f in self.loaded_files if f['active']]
                if active_files:
                     ax.set_title(f"Aktywne pliki z listy ({len(active_files)})" + title_suffix)
                     if self.wavenumbers is not None:
                         # Check if we have preprocessed data matching the list
                         if use_preprocessed and self.preprocessed_tensor is not None:
                             w, k, _ = self.preprocessed_tensor.shape
                             if k == len(active_files):
                                 # Plot preprocessed data
                                 for i, f in enumerate(active_files):
                                     ax.plot(self.wavenumbers, self.preprocessed_tensor[:, i, 0], label=f"{i+1}. {f['name']}")
                             else:
                                 # Fallback to raw if mismatch
                                 for i, f in enumerate(active_files):
                                     ax.plot(self.wavenumbers, f['data'], label=f"{i+1}. {f['name']}")
                         else:
                             # Plot raw data
                             for i, f in enumerate(active_files):
                                 ax.plot(self.wavenumbers, f['data'], label=f"{i+1}. {f['name']}")
                else:
                    ax.set_title("Panel Wykresów (Wybierz komórki lub załaduj pliki)")
            else:
                ax.set_title(f"Zaznaczono {len(self.selected_coords)} widm" + title_suffix)
                if data_source is not None and self.wavenumbers is not None:
                    for coords in self.selected_coords:
                        r, c = coords
                        if not np.isnan(data_source[0, r, c]):
                            ax.plot(self.wavenumbers, data_source[:, r, c], label=f"Widmo {coords}")
            
            if (self.selected_coords and len(self.selected_coords) <= 10) or (not self.selected_coords and len(self.loaded_files) <= 10):
                    legend = ax.legend(loc='best')
                    self._apply_theme_to_legend(legend, theme)

        elif self.current_plot_mode == 'PCA' and self.pca_results:
            for ax in self.axes.flat: ax.set_visible(True)
            scores = self.pca_results['scores'];
            loadings = self.pca_results['loadings']
            variance = self.pca_results['variance'];
            labels = self.pca_results['labels']
            n_components = scores.shape[1]

            ax1 = self.axes[0, 0]  # Score Plot
            ax1.scatter(scores[:, 0], scores[:, 1])
            for i, label in enumerate(labels): ax1.text(scores[i, 0], scores[i, 1], label, fontsize=9,
                                                        color=theme['text_color'])
            ax1.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)");
            ax1.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
            ax1.set_title("Score Plot");
            ax1.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            ax2 = self.axes[0, 1]  # Wspólne Ładunki
            if self.wavenumbers is not None:
                for i in range(n_components): ax2.plot(self.wavenumbers, loadings[:, i], label=f"PC{i + 1}")
            ax2.set_xlabel("Liczba falowa");
            ax2.set_ylabel("Ładunek");
            ax2.set_title("Wspólne Ładunki (Loadings)")
            self._set_wavenumber_axis_inverted(ax2);
            self._apply_theme_to_legend(ax2.legend(loc='best'), theme);
            ax2.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            ax3 = self.axes[1, 0]  # Scores vs. Próbka
            sample_indices = np.arange(len(labels))
            for i in range(n_components): ax3.plot(sample_indices, scores[:, i], 'o-', markersize=4, label=f'PC{i + 1}')
            ax3.set_xlabel("Próbka");
            ax3.set_ylabel("Wartość Wyniku (Score)");
            ax3.set_title("Wyniki (Scores) vs. Próbka")
            ax3.set_xticks(sample_indices);
            ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            self._apply_theme_to_legend(ax3.legend(loc='best'), theme);
            ax3.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            ax4 = self.axes[1, 1]  # Scree Plot
            pc_range = np.arange(1, len(variance) + 1)
            ax4.bar(pc_range, variance * 100, color="#AAAAFF", alpha=0.7)
            ax4.set_ylabel("Wariancja wyjaśniona (%)", color="#AAAAFF");
            ax4.set_xlabel("Komponent")
            ax4.set_title("Scree Plot");
            ax4.set_xticks(pc_range)
            ax4b = ax4.twinx()
            ax4b.plot(pc_range, np.cumsum(variance * 100), 'r-o', markersize=4)
            ax4b.set_ylabel("Suma kumulacyjna (%)", color="red");
            ax4b.tick_params(axis='y', colors="red")
            ax4b.spines['right'].set_color('red')

        elif self.current_plot_mode == 'RECONSTRUCTION' and self.pca_results:
            for ax in self.axes.flat: ax.set_visible(True)
            k = self.pca_results['recon_k'];
            loadings = self.pca_results['loadings']
            X_original = self.pca_results['X_original'];
            X_recon = self.pca_results['X_reconstructed']
            residuals = self.pca_results['residuals']

            ax1 = self.axes[0, 0]  # Wybrane Ładunki
            if self.wavenumbers is not None:
                for i in range(k): ax1.plot(self.wavenumbers, loadings[:, i], label=f"PC{i + 1}")
            ax1.set_title(f"Wybrane Ładunki (Użyte {k} PC)");
            ax1.set_xlabel("Liczba falowa");
            ax1.set_ylabel("Ładunek")
            self._set_wavenumber_axis_inverted(ax1);
            self._apply_theme_to_legend(ax1.legend(loc='best'), theme)

            ax2 = self.axes[0, 1]  # Widma Oryginalne
            if self.wavenumbers is not None:
                for i in range(X_original.shape[0]): ax2.plot(self.wavenumbers, X_original[i, :], alpha=0.7)
            ax2.set_title("Widma Oryginalne");
            ax2.set_xlabel("Liczba falowa");
            ax2.set_ylabel("Absorbancja")
            self._set_wavenumber_axis_inverted(ax2)

            ax3 = self.axes[1, 0]  # Widma Odtworzone
            if self.wavenumbers is not None:
                for i in range(X_recon.shape[0]): ax3.plot(self.wavenumbers, X_recon[i, :], alpha=0.7)
            ax3.set_title(f"Widma Odtworzone (z {k} PC)");
            ax3.set_xlabel("Liczba falowa");
            ax3.set_ylabel("Absorbancja")
            self._set_wavenumber_axis_inverted(ax3)

            ax4 = self.axes[1, 1]  # Rezydua
            if self.wavenumbers is not None:
                for i in range(residuals.shape[0]): ax4.plot(self.wavenumbers, residuals[i, :], alpha=0.7)
            ax4.set_title("Rezydua (Oryginał - Odtworzone)");
            ax4.set_xlabel("Liczba falowa");
            ax4.set_ylabel("Różnica")
            self._set_wavenumber_axis_inverted(ax4)

        elif self.current_plot_mode == 'PARAFAC' and self.parafac_results:
            for ax in self.axes.flat: ax.set_visible(True)
            factors = self.parafac_results['factors']
            weights = self.parafac_results['weights']
            k = factors[0].shape[1]

            ax1 = self.axes[0, 0]  # Widma Składowych (Mode 0)
            if self.wavenumbers is not None:
                for i in range(k): ax1.plot(self.wavenumbers, factors[0][:, i], label=f"Skł. {i + 1}")
            ax1.set_title("PARAFAC: Widma Składowych (Mode 0)")
            ax1.set_xlabel("Liczba falowa");
            ax1.set_ylabel("Ładunek")
            self._set_wavenumber_axis_inverted(ax1);
            self._apply_theme_to_legend(ax1.legend(loc='best'), theme);
            ax1.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            ax2 = self.axes[0, 1]  # Trendy Wierszy (Mode 1)
            x_axis_n = np.arange(self.n_rows)
            for i in range(k): ax2.plot(x_axis_n, factors[1][:, i], 'o-', markersize=4, label=f"Skł. {i + 1}")
            ax2.set_title("PARAFAC: Trendy Wierszy (Mode 1)")
            ax2.set_xlabel("Indeks Wiersza (N)");
            ax2.set_ylabel("Waga")
            ax2.set_xticks(x_axis_n)
            self._apply_theme_to_legend(ax2.legend(loc='best'), theme);
            ax2.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            ax3 = self.axes[1, 0]  # Trendy Kolumn (Mode 2)
            x_axis_m = np.arange(self.m_cols)
            for i in range(k): ax3.plot(x_axis_m, factors[2][:, i], 'o-', markersize=4, label=f"Skł. {i + 1}")
            ax3.set_title("PARAFAC: Trendy Kolumn (Mode 2)")
            ax3.set_xlabel("Indeks Kolumny (M)");
            ax3.set_ylabel("Waga")
            ax3.set_xticks(x_axis_m)
            self._apply_theme_to_legend(ax3.legend(loc='best'), theme);
            ax3.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            ax4 = self.axes[1, 1]  # Wagi Składowych
            comp_range = np.arange(1, k + 1)
            ax4.bar(comp_range, weights, color="#AAAAFF", alpha=0.7)
            ax4.set_ylabel("Waga Składowej", color=theme['text_color'])
            ax4.set_xlabel("Składowa");
            ax4.set_title("PARAFAC: Wagi Składowych")
            ax4.set_xticks(comp_range)

        elif self.current_plot_mode == 'MCR' and self.mcr_results:
            for ax in self.axes.flat: ax.set_visible(True)
            C = self.mcr_results['C']  # (n*m, k)
            ST = self.mcr_results['ST']  # (w, k)
            k = self.mcr_results['rank']

            ax1 = self.axes[0, 0]  # Widma Składowych
            if self.wavenumbers is not None:
                for i in range(k): ax1.plot(self.wavenumbers, ST[:, i], label=f"Skł. {i + 1}")
            ax1.set_title("MCR: Widma Składowych");
            ax1.set_xlabel("Liczba falowa");
            ax1.set_ylabel("Intensywność")
            self._set_wavenumber_axis_inverted(ax1);
            self._apply_theme_to_legend(ax1.legend(loc='best'), theme);
            ax1.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            ax2 = self.axes[0, 1]  # Profile Stężeń (Liniowo)
            x_axis_nm = np.arange(C.shape[0])
            for i in range(k): ax2.plot(x_axis_nm, C[:, i], 'o-', markersize=3, alpha=0.8, label=f"Skł. {i + 1}")
            ax2.set_title("MCR: Profile Stężeń (Liniowo)");
            ax2.set_xlabel(f"Indeks Próbki (0...{len(x_axis_nm) - 1})");
            ax2.set_ylabel("Względne Stężenie")
            self._apply_theme_to_legend(ax2.legend(loc='best'), theme);
            ax2.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            ax3 = self.axes[1, 0]  # Mapa Stężeń (Skł. 1)
            ax4 = self.axes[1, 1]  # Mapa Stężeń (Skł. 2)

            is_grid = self.mcr_results.get('is_grid_data', False)
            
            if is_grid and self.n_rows * self.m_cols == C.shape[0]:
                c1_map = C[:, 0].reshape((self.n_rows, self.m_cols))
                im3 = ax3.imshow(c1_map, aspect='auto', interpolation='nearest', cmap='viridis')
                self.fig.colorbar(im3, ax=ax3)
                ax3.set_title("MCR: Mapa Stężeń (Skł. 1)");
                ax3.set_xlabel("Indeks Kolumny (M)");
                ax3.set_ylabel("Indeks Wiersza (N)")

                if k > 1:
                    c2_map = C[:, 1].reshape((self.n_rows, self.m_cols))
                    im4 = ax4.imshow(c2_map, aspect='auto', interpolation='nearest', cmap='viridis')
                    self.fig.colorbar(im4, ax=ax4)
                    ax4.set_title("MCR: Mapa Stężeń (Skł. 2)")
                else:
                    ax4.set_title("MCR")
                ax4.set_xlabel("Indeks Kolumny (M)");
                ax4.set_ylabel("Indeks Wiersza (N)")
            else:
                # List mode - plot concentration profiles as bar/line plot instead of map
                ax3.plot(C[:, 0], 'o-', label="Skł. 1")
                ax3.set_title("MCR: Profil Stężeń (Skł. 1)")
                ax3.set_xlabel("Indeks Próbki (Lista)")
                ax3.set_ylabel("Stężenie")
                ax3.grid(True, linestyle=':', alpha=0.5)
                
                if k > 1:
                    ax4.plot(C[:, 1], 'o-', label="Skł. 2", color='orange')
                    ax4.set_title("MCR: Profil Stężeń (Skł. 2)")
                else:
                    ax4.set_title("MCR")
                ax4.set_xlabel("Indeks Próbki (Lista)")
                ax4.set_ylabel("Stężenie")
                ax4.grid(True, linestyle=':', alpha=0.5)

        elif self.current_plot_mode == 'TENSOR_RECONSTRUCTION' and self.tensor_recon_results:
            for ax in self.axes.flat: ax.set_visible(True)
            k = self.tensor_recon_results['k']
            source = self.tensor_recon_results['source']
            components = self.tensor_recon_results['components']  # (w, k)
            source_tensor = self.tensor_recon_results['source_tensor']  # (w, n, m)
            recon_tensor = self.tensor_recon_results['recon_tensor']
            resid_tensor = self.tensor_recon_results['residual_tensor']

            ax1 = self.axes[0, 0]  # Wybrane Ładunki
            if self.wavenumbers is not None:
                for i in range(k): ax1.plot(self.wavenumbers, components[:, i], label=f"Skł. {i + 1}")
            ax1.set_title(f"Użyte Składowe ({source.upper()}, k={k})");
            ax1.set_xlabel("Liczba falowa");
            ax1.set_ylabel("Intensywność")
            self._set_wavenumber_axis_inverted(ax1);
            self._apply_theme_to_legend(ax1.legend(loc='best'), theme)

            ax2 = self.axes[0, 1]  # Widma Oryginalne (z zaznaczenia)
            if not self.selected_coords:
                ax2.text(0.5, 0.5, "Zaznacz komórki\naby zobaczyć podgląd", ha='center', va='center',
                         color=theme['text_color'])
            elif self.wavenumbers is not None:
                for coords in self.selected_coords:
                    r, c = coords
                    ax2.plot(self.wavenumbers, source_tensor[:, r, c], alpha=0.7)
            ax2.set_title("Widma Oryginalne (z zaznaczenia)");
            ax2.set_xlabel("Liczba falowa");
            ax2.set_ylabel("Absorbancja")
            self._set_wavenumber_axis_inverted(ax2)

            ax3 = self.axes[1, 0]  # Widma Odtworzone (z zaznaczenia)
            if not self.selected_coords:
                ax3.text(0.5, 0.5, "Zaznacz komórki\naby zobaczyć podgląd", ha='center', va='center',
                         color=theme['text_color'])
            elif self.wavenumbers is not None:
                for coords in self.selected_coords:
                    r, c = coords
                    ax3.plot(self.wavenumbers, recon_tensor[:, r, c], alpha=0.7)
            ax3.set_title("Widma Odtworzone (z zaznaczenia)");
            ax3.set_xlabel("Liczba falowa");
            ax3.set_ylabel("Absorbancja")
            self._set_wavenumber_axis_inverted(ax3)

            ax4 = self.axes[1, 1]  # Rezydua (z zaznaczenia)
            if not self.selected_coords:
                ax4.text(0.5, 0.5, "Zaznacz komórki\naby zobaczyć podgląd", ha='center', va='center',
                         color=theme['text_color'])
            elif self.wavenumbers is not None:
                for coords in self.selected_coords:
                    r, c = coords
                    ax4.plot(self.wavenumbers, resid_tensor[:, r, c], alpha=0.7)
            ax4.set_title("Rezydua (Oryginał - Odtworzone)");
            ax4.set_xlabel("Liczba falowa");
            ax4.set_ylabel("Różnica")
            self._set_wavenumber_axis_inverted(ax4)

        elif self.current_plot_mode == '3DCOS_SLICER' and self.cos_3d_results:
            self.cos_slider_label.grid()
            self.cos_slider.grid()

            slice_index = int(self.cos_slice_var.get())
            self.cos_slider_label.configure(text=f"Plaster Modulatora ({slice_index}):")

            phi_map = self.cos_3d_results['phi'][:, :, slice_index]
            psi_map = self.cos_3d_results['psi'][:, :, slice_index]

            v_min_ax = np.min(self.wavenumbers)
            v_max_ax = np.max(self.wavenumbers)
            extent = [v_max_ax, v_min_ax, v_max_ax, v_min_ax]

            ax1 = self.axes[0, 0]
            ax1.set_visible(True)
            vmax_phi = np.max(np.abs(phi_map[np.isfinite(phi_map)]))
            im1 = ax1.imshow(phi_map, cmap='RdBu_r', vmin=-vmax_phi, vmax=vmax_phi, extent=extent,
                             interpolation='nearest')
            ax1.set_title(f"Mapa Synchroniczna (Plaster {slice_index})")
            ax1.set_xlabel("Liczba falowa (v1)");
            ax1.set_ylabel("Liczba falowa (v2)")
            self.fig.colorbar(im1, ax=ax1, format='%.2e')

            ax2 = self.axes[0, 1]
            ax2.set_visible(True)
            vmax_psi = np.max(np.abs(psi_map[np.isfinite(psi_map)]))
            im2 = ax2.imshow(psi_map, cmap='RdBu_r', vmin=-vmax_psi, vmax=vmax_psi, extent=extent,
                             interpolation='nearest')
            ax2.set_title(f"Mapa Asynchroniczna (Plaster {slice_index})")
            ax2.set_xlabel("Liczba falowa (v1)");
            ax2.set_ylabel("Liczba falowa (v2)")
            self.fig.colorbar(im2, ax=ax2, format='%.2e')

            self.axes[1, 0].set_visible(False)
            self.axes[1, 1].set_visible(False)

        elif self.current_plot_mode == 'PLS_RESULTS' and self.pls_results:
            ax = self.fig.get_axes()[0]
            
            coefs = self.pls_results['coefs']
            target_name = self.pls_results['target_name']

            if self.wavenumbers is not None:
                ax.plot(self.wavenumbers, coefs)
                ax.set_xlabel("Liczba falowa")
                self._set_wavenumber_axis_inverted(ax)
            else:
                ax.plot(coefs)
                ax.set_xlabel("Indeks Zmiennej")

            ax.set_ylabel("Współczynnik Regresji PLS")
            ax.set_title(f"Ważność Zmiennych dla: {target_name}")
            ax.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])
            ax.axhline(0, color=theme['spine_color'], linestyle='--')

        elif self.current_plot_mode == 'TUCKER_RESULTS' and self.tucker_results:
            for ax in self.axes.flat: ax.set_visible(True)
            core = self.tucker_results['core']  # (rw, rn, rm)
            factors = self.tucker_results['factors']  # [(w, rw), (n, rn), (m, rm)]

            rank_w, rank_n, rank_m = core.shape

            ax1 = self.axes[0, 0]  # Faktor 0 (Widma)
            if self.wavenumbers is not None:
                for i in range(rank_w): ax1.plot(self.wavenumbers, factors[0][:, i], label=f"Skł. {i + 1}")
            ax1.set_title("Tucker: Widma Bazowe (Mode 0)")
            ax1.set_xlabel("Liczba falowa");
            ax1.set_ylabel("Ładunek")
            self._set_wavenumber_axis_inverted(ax1);
            self._apply_theme_to_legend(ax1.legend(loc='best'), theme)

            ax2 = self.axes[0, 1]  # Faktor 1 (Trendy N)
            x_axis_n = np.arange(self.n_rows)
            for i in range(rank_n): ax2.plot(x_axis_n, factors[1][:, i], 'o-', markersize=4, label=f"Skł. {i + 1}")
            ax2.set_title("Tucker: Trendy Wierszy (Mode 1)")
            ax2.set_xlabel("Indeks Wiersza (N)");
            ax2.set_ylabel("Waga")
            ax2.set_xticks(x_axis_n)
            self._apply_theme_to_legend(ax2.legend(loc='best'), theme)

            ax3 = self.axes[1, 0]  # Faktor 2 (Trendy M)
            x_axis_m = np.arange(self.m_cols)
            for i in range(rank_m): ax3.plot(x_axis_m, factors[2][:, i], 'o-', markersize=4, label=f"Skł. {i + 1}")
            ax3.set_title("Tucker: Trendy Kolumn (Mode 2)")
            ax3.set_xlabel("Indeks Kolumny (M)");
            ax3.set_ylabel("Waga")
            ax3.set_xticks(x_axis_m)
            self._apply_theme_to_legend(ax3.legend(loc='best'), theme)

            ax4 = self.axes[1, 1]  # Tensor Rdzeniowy (Plaster 0)
            core_slice = core[:, :, 0]
            vmax = np.max(np.abs(core_slice))
            im4 = ax4.imshow(core_slice, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest', aspect='auto')
            ax4.set_title(f"Tensor Rdzeniowy $\mathcal{{G}}$ (Plaster: 0)")
            ax4.set_xlabel(f"Indeks Wiersza (N) [0..{rank_n - 1}]");
            ax4.set_ylabel(f"Indeks Widma (W) [0..{rank_w - 1}]")
            ax4.set_xticks(np.arange(rank_n));
            ax4.set_yticks(np.arange(rank_w))
            self.fig.colorbar(im4, ax=ax4)

        elif self.current_plot_mode == 'FA_RANK_RESULTS' and self.fa_results:
            for ax in self.axes.flat: ax.set_visible(True)
            ev = self.fa_results['ev'];
            re = self.fa_results['re'];
            ind = self.fa_results['ind']
            max_k = self.fa_results['max_k'];
            x_axis = np.arange(1, max_k + 1)
            loadings = self.fa_results['vh'].T  # (cechy, k)
            scores_raw = self.fa_results['u'] @ np.diag(self.fa_results['s_vec'])  # (próbki, k)
            labels = self.fa_results['labels']

            ax1 = self.axes[0, 0]  # Wykres Osypiska (EV)
            ax1.plot(x_axis, ev[:max_k], 'o-', label='Wartości Własne')
            ax1.set_yscale('log')
            ax1.set_title("Wykres Osypiska (Eigenvalues)")
            ax1.set_xlabel("Liczba Faktorów");
            ax1.set_ylabel("Log(Wartość Własna)")
            ax1.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            ax2 = self.axes[0, 1]  # Błąd Rzeczywisty (RE) i Funkcja Wskaźnikowa (IND)
            color_re = 'tab:blue'
            ax2.plot(x_axis, np.log(re), 'o-', color=color_re, label='log(RE)')
            ax2.set_xlabel("Liczba Faktorów");
            ax2.set_ylabel("log(RE)", color=color_re)
            ax2.tick_params(axis='y', labelcolor=color_re)
            ax2.set_title("Wskaźniki Rangi Malinowskiego")

            ax2b = ax2.twinx()
            color_ind = 'tab:red'
            ax2b.plot(x_axis, ind, 'o-', color=color_ind, label='IND')
            ax2b.set_ylabel("IND", color=color_ind)
            ax2b.tick_params(axis='y', labelcolor=color_ind)
            ax2.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            ax3 = self.axes[1, 0]  # Widma Abstrakcyjne
            if self.wavenumbers is not None:
                for i in range(max_k): ax3.plot(self.wavenumbers, loadings[:, i], label=f"Faktor {i + 1}")
            ax3.set_title("Widma Abstrakcyjne (Ładunki)");
            ax3.set_xlabel("Liczba falowa")
            self._set_wavenumber_axis_inverted(ax3);
            if max_k <= 10: self._apply_theme_to_legend(ax3.legend(loc='best'), theme)

            ax4 = self.axes[1, 1]  # Profile Abstrakcyjne
            sample_indices = np.arange(len(labels))
            for i in range(max_k): ax4.plot(sample_indices, scores_raw[:, i], 'o-', markersize=4,
                                            label=f'Faktor {i + 1}')
            ax4.set_title("Profile Abstrakcyjne (Wyniki)");
            ax4.set_xlabel("Próbka")
            ax4.set_xticks(sample_indices);
            ax4.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            if max_k <= 10: self._apply_theme_to_legend(ax4.legend(loc='best'), theme)

        elif self.current_plot_mode == 'FA_RECON_RESULTS' and self.fa_results:
            for ax in self.axes.flat: ax.set_visible(True)
            k = self.fa_results['recon_k']
            loadings_k = self.fa_results['recon_loadings_k']  # (cechy, k)
            X_original = self.fa_results['recon_X_original']
            X_recon = self.fa_results['recon_X_recon']
            residuals = self.fa_results['recon_residuals']

            ax1 = self.axes[0, 0]  # Wybrane Ładunki
            if self.wavenumbers is not None:
                for i in range(k): ax1.plot(self.wavenumbers, loadings_k[:, i], label=f"Faktor {i + 1}")
            ax1.set_title(f"Użyte Faktory (Widma Abstrakcyjne, k={k})");
            ax1.set_xlabel("Liczba falowa");
            ax1.set_ylabel("Ładunek")
            self._set_wavenumber_axis_inverted(ax1);
            self._apply_theme_to_legend(ax1.legend(loc='best'), theme)

            ax2 = self.axes[0, 1]  # Widma Oryginalne
            if self.wavenumbers is not None:
                for i in range(X_original.shape[0]): ax2.plot(self.wavenumbers, X_original[i, :], alpha=0.7)
            ax2.set_title("Widma Oryginalne (Zaznaczone)");
            ax2.set_xlabel("Liczba falowa");
            ax2.set_ylabel("Absorbancja")
            self._set_wavenumber_axis_inverted(ax2)

            ax3 = self.axes[1, 0]  # Widma Odtworzone
            if self.wavenumbers is not None:
                for i in range(X_recon.shape[0]): ax3.plot(self.wavenumbers, X_recon[i, :], alpha=0.7)
            ax3.set_title(f"Widma Odtworzone (z {k} faktorów)");
            ax3.set_xlabel("Liczba falowa");
            ax3.set_ylabel("Absorbancja")
            self._set_wavenumber_axis_inverted(ax3)

            ax4 = self.axes[1, 1]  # Rezydua
            if self.wavenumbers is not None:
                for i in range(residuals.shape[0]): ax4.plot(self.wavenumbers, residuals[i, :], alpha=0.7)
            ax4.set_title("Rezydua (Oryginał - Odtworzone)");
            ax4.set_xlabel("Liczba falowa");
            ax4.set_ylabel("Różnica")
            self._set_wavenumber_axis_inverted(ax4)

        elif self.current_plot_mode == 'SPEXFA_RESULTS' and self.spexfa_results:
            ax1 = self.axes[0, 0]  # Widma Wyizolowane
            ax1.set_visible(True)
            ax2 = self.axes[0, 1]  # Profile Stężeń
            ax2.set_visible(True)
            self.axes[1, 0].set_visible(False)
            self.axes[1, 1].set_visible(False)

            C = self.spexfa_results['C']  # (próbki, k)
            ST = self.spexfa_results['ST']  # (cechy, k)
            k = self.spexfa_results['rank']
            labels = self.spexfa_results['labels']

            if self.wavenumbers is not None:
                for i in range(k): ax1.plot(self.wavenumbers, ST[:, i], label=f"Skł. {i + 1}")
            ax1.set_title("SPEXFA: Widma Wyizolowane");
            ax1.set_xlabel("Liczba falowa");
            ax1.set_ylabel("Intensywność")
            self._set_wavenumber_axis_inverted(ax1);
            self._apply_theme_to_legend(ax1.legend(loc='best'), theme);
            ax1.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

            sample_indices = np.arange(len(labels))
            for i in range(k): ax2.plot(sample_indices, C[:, i], 'o-', markersize=4, label=f"Skł. {i + 1}")
            ax2.set_title("SPEXFA: Profile Stężeń");
            ax2.set_xlabel("Próbka");
            ax2.set_ylabel("Względne Stężenie")
            ax2.set_xticks(sample_indices);
            ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            self._apply_theme_to_legend(ax2.legend(loc='best'), theme);
            ax2.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

        elif self.current_plot_mode == 'HEATMAP':
            ax = self.fig.get_axes()[0]
            
            data_source = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
            if data_source is not None:
                # Suma po osi spektralnej (prosta wizualizacja)
                heatmap_data = np.nansum(data_source, axis=0)
                im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
                self.fig.colorbar(im, ax=ax)
                ax.set_title("Mapa Ciepła (Suma Absorbancji)")
                ax.set_xlabel("Indeks Kolumny (M)")
                ax.set_ylabel("Indeks Wiersza (N)")
            else:
                ax.text(0.5, 0.5, "Brak danych do mapy ciepła", ha='center', va='center')

        elif self.current_plot_mode == 'MANIFOLD_PLOT' and self.manifold_results:
            ax = self.fig.get_axes()[0]
            
            scores = self.manifold_results['scores']
            labels = self.manifold_results['labels']
            method = self.manifold_results['method_name']

            ax.scatter(scores[:, 0], scores[:, 1])
            for i, label in enumerate(labels):
                ax.text(scores[i, 0], scores[i, 1], label, fontsize=9, color=theme['text_color'])
            ax.set_xlabel("Komponent 1");
            ax.set_ylabel("Komponent 2")
            ax.set_title(f"Wykres Wyników {method}");
            ax.grid(True, linestyle=':', alpha=0.2, color=theme['grid_color'])

        # Hide unused axes for 2x2 layouts
        if self.current_plot_mode not in single_plot_modes:
            for ax in self.axes.flat:
                if not ax.get_children(): # Check if axis is empty
                    ax.set_visible(False)
        
        # Store initial limits for all visible axes
        self.initial_lims.clear()
        for ax in self.fig.get_axes():
            if ax.get_visible():
                self.initial_lims[ax] = (ax.get_xlim(), ax.get_ylim())

        self.fig.tight_layout()
        self.canvas.draw()

    def _on_canvas_button_press(self, event):
        # Fix for focus issue: ensure canvas takes focus so buttons don't re-trigger on Enter/Space
        self.canvas.get_tk_widget().focus_set()

        if event.inaxes is None: return
        if self.toolbar.mode != "": return

        # Find which axis was clicked
        for ax in self.fig.get_axes():
            if ax == event.inaxes and ax.get_visible():
                self.zoom_start[ax] = (event.xdata, event.ydata)
                # Save initial limits if not saved
                if ax not in self.initial_lims:
                    self.initial_lims[ax] = (ax.get_xlim(), ax.get_ylim())
                break

    def _on_canvas_motion(self, event):
        if event.inaxes is None or self.toolbar.mode != "": return

        for ax in self.fig.get_axes():
            if ax == event.inaxes and ax in self.zoom_start and self.zoom_start[ax] is not None:
                x_start, y_start = self.zoom_start[ax]
                x_curr, y_curr = event.xdata, event.ydata
                
                # Draw rectangle (remove old one first)
                if ax in self.zoom_rects and self.zoom_rects[ax] is not None:
                    self.zoom_rects[ax].remove()
                    self.zoom_rects[ax] = None
                
                rect = matplotlib.patches.Rectangle((min(x_start, x_curr), min(y_start, y_curr)),
                                                    abs(x_curr - x_start), abs(y_curr - y_start),
                                                    linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                self.zoom_rects[ax] = rect
                self.canvas.draw_idle()
                break

    def _on_canvas_button_release(self, event):
        if event.inaxes is None or self.toolbar.mode != "": return

        for ax in self.fig.get_axes():
            if ax == event.inaxes and ax in self.zoom_start and self.zoom_start[ax] is not None:
                x_start, y_start = self.zoom_start[ax]
                x_curr, y_curr = event.xdata, event.ydata
                
                self.zoom_start[ax] = None
                if ax in self.zoom_rects and self.zoom_rects[ax] is not None:
                    self.zoom_rects[ax].remove()
                    self.zoom_rects[ax] = None
                
                # If click (small movement), reset zoom
                # Use abs() for range to handle inverted axes
                if abs(x_curr - x_start) < abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01 and \
                   abs(y_curr - y_start) < abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01:
                       if ax in self.initial_lims:
                           ax.set_xlim(self.initial_lims[ax][0])
                           ax.set_ylim(self.initial_lims[ax][1])
                           self.canvas.draw_idle()
                else:
                    # Apply zoom
                    # Check if axes are inverted before setting new limits
                    x_a, x_b = sorted([x_start, x_curr])
                    y_a, y_b = sorted([y_start, y_curr])

                    if ax.xaxis_inverted():
                        ax.set_xlim(x_b, x_a) # Keep inverted (High -> Low)
                    else:
                        ax.set_xlim(x_a, x_b) # Standard (Low -> High)

                    if ax.yaxis_inverted():
                        ax.set_ylim(y_b, y_a)
                    else:
                        ax.set_ylim(y_a, y_b)

                    self.canvas.draw_idle()
                break

    def _set_window_size(self):
        pass

    def _save_project(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Pliki Projektu ChemTensor", "*.json"), ("Wszystkie pliki", "*.*")]
        )
        if not filepath:
            return

        try:
            status_to_save = {f"{r},{c}": s for (r, c), s in self.field_matrix_status.items()}

            project_data = {
                'n_rows': self.n_rows,
                'm_cols': self.m_cols,
                'original_wavenumbers': self.original_wavenumbers.tolist() if self.original_wavenumbers is not None else None,
                'original_tensor_data': np.where(np.isnan(self.original_tensor_data), None,
                                                 self.original_tensor_data).tolist() if self.original_tensor_data is not None else None,
                'field_matrix_status': status_to_save
            }

            with open(filepath, 'w') as f:
                json.dump(project_data, f, indent=4)
            messagebox.showinfo("Sukces", f"Projekt został pomyślnie zapisany w:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Błąd Zapisu", f"Nie udało się zapisać projektu.\n{e}")

    def _load_project(self):
        filepath = filedialog.askopenfilename(
            title="Wybierz plik projektu",
            filetypes=[("Pliki Projektu ChemTensor", "*.json"), ("Wszystkie pliki", "*.*")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                project_data = json.load(f)

            self.n_var.set(str(project_data['n_rows']))
            self.m_var.set(str(project_data['m_cols']))
            self._create_field_matrix()

            if project_data['original_wavenumbers'] is not None:
                self.original_wavenumbers = np.array(project_data['original_wavenumbers'])
            if project_data['original_tensor_data'] is not None:
                self.original_tensor_data = np.array(project_data['original_tensor_data'], dtype=float)

            self._reset_wavenumber_range()

            loaded_status_map = project_data['field_matrix_status']
            for str_coords, status in loaded_status_map.items():
                r, c = map(int, str_coords.split(','))
                coords = (r, c)
                self.field_matrix_status[coords] = status
                self._apply_visuals_for_status(coords, status)

            self.update_plot()
            messagebox.showinfo("Sukces", f"Projekt został pomyślnie wczytany z:\n{filepath}")
            self.focus_set()

        except Exception as e:
            messagebox.showerror("Błąd Odczytu", f"Nie udało się wczytać projektu.\n{e}")
            self._create_field_matrix()

    def _export_to_xlsx(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Plik Excel", "*.xlsx"), ("Wszystkie pliki", "*.*")]
        )
        if not filepath:
            return

        print("Rozpoczynanie eksportu do XLSX...")
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                if self.wavenumbers is not None:
                    pd.DataFrame(self.wavenumbers, columns=['Wavenumber']).to_excel(writer,
                                                                                    sheet_name='Wavenumbers_Active',
                                                                                    index=False)

                if self.tensor_data is not None:
                    data_2d = self.tensor_data.transpose(1, 2, 0).reshape((self.n_rows * self.m_cols), -1).T
                    cols = [f"({r},{c})" for r in range(self.n_rows) for c in range(self.m_cols)]
                    df = pd.DataFrame(data_2d, columns=cols, index=self.wavenumbers)
                    df.index.name = "Wavenumber"
                    df.to_excel(writer, sheet_name='Active_Data')

                if self.preprocessed_tensor is not None:
                    data_2d_proc = self.preprocessed_tensor.transpose(1, 2, 0).reshape((self.n_rows * self.m_cols),
                                                                                       -1).T
                    cols_proc = [f"({r},{c})" for r in range(self.n_rows) for c in range(self.m_cols)]
                    df_proc = pd.DataFrame(data_2d_proc, columns=cols_proc, index=self.wavenumbers)
                    df_proc.index.name = "Wavenumber"
                    df_proc.to_excel(writer, sheet_name='Preprocessed_Data')

                if self.pca_results is not None:
                    pd.DataFrame(self.pca_results['scores'],
                                 columns=[f"PC{i + 1}" for i in range(self.pca_results['scores'].shape[1])],
                                 index=self.pca_results['labels']).to_excel(writer, sheet_name='PCA_Scores')
                    pd.DataFrame(self.pca_results['loadings'],
                                 columns=[f"PC{i + 1}" for i in range(self.pca_results['loadings'].shape[1])],
                                 index=self.wavenumbers).to_excel(writer, sheet_name='PCA_Loadings',
                                                                  index_label="Wavenumber")
                    pd.DataFrame(self.pca_results['variance'],
                                 columns=['Explained_Variance']).to_excel(writer, sheet_name='PCA_Variance')

                if self.parafac_results is not None:
                    factors = self.parafac_results['factors']
                    k = factors[0].shape[1]
                    pd.DataFrame(factors[0], index=self.wavenumbers,
                                 columns=[f"Comp {i + 1}" for i in range(k)]).to_excel(writer,
                                                                                       sheet_name='PARAFAC_Spectra',
                                                                                       index_label="Wavenumber")
                    pd.DataFrame(factors[1], index=[f"Row {i}" for i in range(self.n_rows)],
                                 columns=[f"Comp {i + 1}" for i in range(k)]).to_excel(writer,
                                                                                       sheet_name='PARAFAC_Trends_N')
                    pd.DataFrame(factors[2], index=[f"Col {i}" for i in range(self.m_cols)],
                                 columns=[f"Comp {i + 1}" for i in range(k)]).to_excel(writer,
                                                                                       sheet_name='PARAFAC_Trends_M')
                    pd.DataFrame(self.parafac_results['weights'],
                                 columns=['Weight']).to_excel(writer, sheet_name='PARAFAC_Weights')

                if self.tucker_results is not None:
                    factors = self.tucker_results['factors']
                    core = self.tucker_results['core']
                    r_w, r_n, r_m = core.shape

                    pd.DataFrame(factors[0], index=self.wavenumbers,
                                 columns=[f"Comp_W {i + 1}" for i in range(r_w)]).to_excel(writer,
                                                                                           sheet_name='Tucker_Spectra_W',
                                                                                           index_label="Wavenumber")
                    pd.DataFrame(factors[1], index=[f"Row {i}" for i in range(self.n_rows)],
                                 columns=[f"Comp_N {i + 1}" for i in range(r_n)]).to_excel(writer,
                                                                                           sheet_name='Tucker_Trends_N')
                    pd.DataFrame(factors[2], index=[f"Col {i}" for i in range(self.m_cols)],
                                 columns=[f"Comp_M {i + 1}" for i in range(r_m)]).to_excel(writer,
                                                                                           sheet_name='Tucker_Trends_M')

                    for i in range(r_m):
                        df_core_slice = pd.DataFrame(core[:, :, i],
                                                     index=[f"W_Comp {j + 1}" for j in range(r_w)],
                                                     columns=[f"N_Comp {j + 1}" for j in range(r_n)])
                        df_core_slice.to_excel(writer, sheet_name=f'Tucker_Core_Slice_M{i + 1}')

                if self.mcr_results is not None:
                    cols_c = [f"({r},{c})" for r in range(self.n_rows) for c in range(self.m_cols)]
                    pd.DataFrame(self.mcr_results['C'], index=cols_c,
                                 columns=[f"Comp {i + 1}" for i in range(self.mcr_results['rank'])]).to_excel(writer,
                                                                                                              sheet_name='MCR_Concentrations')
                    pd.DataFrame(self.mcr_results['ST'], index=self.wavenumbers,
                                 columns=[f"Comp {i + 1}" for i in range(self.mcr_results['rank'])]).to_excel(writer,
                                                                                                              sheet_name='MCR_Spectra',
                                                                                                              index_label="Wavenumber")

                if self.fa_results is not None:
                    if 'ev' in self.fa_results:
                        pd.DataFrame(self.fa_results['ev'], columns=['Eigenvalue']).to_excel(writer,
                                                                                             sheet_name='FA_Eigenvalues')
                    if 're' in self.fa_results:
                        pd.DataFrame({'RE': self.fa_results['re'], 'IND': self.fa_results['ind']}).to_excel(writer,
                                                                                                            sheet_name='FA_Indicators')
                    if 'recon_loadings_k' in self.fa_results:
                        pd.DataFrame(self.fa_results['recon_loadings_k'], index=self.wavenumbers,
                                     columns=[f"Factor {i + 1}" for i in range(self.fa_results['recon_k'])]).to_excel(
                            writer, sheet_name='FA_Abstract_Spectra', index_label="Wavenumber")
                    if 'recon_scores_k' in self.fa_results:
                        cols_fa_scores = self.fa_results['labels']
                        pd.DataFrame(self.fa_results['recon_scores_k'], index=cols_fa_scores,
                                     columns=[f"Factor {i + 1}" for i in range(self.fa_results['recon_k'])]).to_excel(
                            writer, sheet_name='FA_Abstract_Profiles')

                if self.spexfa_results is not None:
                    cols_c = self.spexfa_results['labels']
                    pd.DataFrame(self.spexfa_results['C'], index=cols_c,
                                 columns=[f"Comp {i + 1}" for i in range(self.spexfa_results['rank'])]).to_excel(writer,
                                                                                                                 sheet_name='SPEXFA_Concentrations')
                    pd.DataFrame(self.spexfa_results['ST'], index=self.wavenumbers,
                                 columns=[f"Comp {i + 1}" for i in range(self.spexfa_results['rank'])]).to_excel(writer,
                                                                                                                 sheet_name='SPEXFA_Spectra',
                                                                                                                 index_label="Wavenumber")

                if self.manifold_results is not None:
                    pd.DataFrame(self.manifold_results['scores'],
                                 columns=["Comp 1", "Comp 2"],
                                 index=self.manifold_results['labels']).to_excel(writer,
                                                                                 sheet_name=f"{self.manifold_results['method_name']}_Scores")

            messagebox.showinfo("Sukces", f"Wyniki zostały pomyślnie wyeksportowane do:\n{filepath}")
        except ImportError:
            messagebox.showerror("Błąd Importu",
                                 "Nie znaleziono biblioteki 'openpyxl' lub 'umap-learn'.\nUpewnij się, że są zainstalowane.")
        except Exception as e:
            messagebox.showerror("Błąd Eksportu", f"Nie udało się wyeksportować pliku.\n{e}")

    def _export_figure(self, file_format):
        pass


if __name__ == "__main__":
    app = ChemTensorApp()
    app.mainloop()