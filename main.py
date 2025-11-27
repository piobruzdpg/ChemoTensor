import tkinter
from tkinter import filedialog, messagebox, Menu
import customtkinter as ctk
import matplotlib.figure
import matplotlib.backends.backend_tkagg
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

# --- IMPORTY NASZYCH MODUŁÓW ---
from modules import preprocessing as prep
from modules import analysis as anal
from modules import data_io
from modules import visualization as vis


# --- Stałe Kolorów Statusu ---
STATUS_COLORS = {
    'EMPTY': ctk.ThemeManager.theme["CTkButton"]["fg_color"], 
    'MISSING': "gray50", 
    'LOADED': "#2ECC71",
    'ERROR': "#E74C3C", 
    'FILLED': "#0096FF"
}
SELECTED_BORDER_COLOR = "#3498DB"

class ChemTensorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ChemTensor Explorer (Modular)")
        self.geometry("1400x1000")

        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

        # Zmienne GUI
        self.n_var = ctk.StringVar(value="4")
        self.m_var = ctk.StringVar(value="4")
        self.range_min_var = ctk.StringVar()
        self.range_max_var = ctk.StringVar()
        
        # Preprocessing vars
        self.sg_window_var = ctk.StringVar(value="11")
        self.sg_poly_var = ctk.StringVar(value="3")
        self.sg_smooth_window_var = ctk.StringVar(value="11")
        self.sg_smooth_poly_var = ctk.StringVar(value="3")
        self.als_lambda_var = ctk.StringVar(value="1e6")
        self.als_p_var = ctk.StringVar(value="0.01")
        self.show_preprocessed_var = ctk.BooleanVar(value=False)
        self.pipeline_steps = []  # Lista słowników: {'type': 'SNV', 'params': {...}}

        # Analysis vars
        self.pca_n_components_var = ctk.StringVar(value="2")
        self.pca_recon_components_var = ctk.StringVar(value="2")
        self.fa_recon_components_var = ctk.StringVar(value="2")
        self.parafac_rank_var = ctk.StringVar(value="2")
        self.parafac_non_negative_var = ctk.BooleanVar(value=False)
        self.mcr_n_components_var = ctk.StringVar(value="2")
        self.mcr_non_negative_var = ctk.BooleanVar(value=True)
        self.mcr_norm_var = ctk.BooleanVar(value=False)
        self.mcr_st_fix_var = ctk.StringVar(value="")
        self.tucker_rank_w_var = ctk.StringVar(value="2")
        self.tucker_rank_n_var = ctk.StringVar(value="2")
        self.tucker_rank_m_var = ctk.StringVar(value="2")
        self.tensor_recon_components_var = ctk.StringVar(value="2")
        self.spexfa_n_components_var = ctk.StringVar(value="2")
        self.pls_target_var = ctk.StringVar(value="Wybierz Cel (y)...")
        self.cos_axis_var = ctk.StringVar(value="Analizuj Wiersze (N)")
        self.cos_slice_var = ctk.DoubleVar(value=0)

        # Stan Danych
        self.field_matrix_widgets = {}
        self.field_matrix_status = {}
        self.selected_coords = set()
        
        self.original_tensor_data = None
        self.original_wavenumbers = None
        self.tensor_data = None      # Dane robocze (po cropie)
        self.wavenumbers = None      # Oś X (po cropie)
        self.preprocessed_tensor = None

        # Wyniki analiz (słowniki)
        self.pca_results = None
        self.parafac_results = None
        self.mcr_results = None
        self.tucker_results = None
        self.spexfa_results = None
        self.pls_results = None
        self.tensor_recon_results = None
        self.cos_3d_results = None
        self.manifold_results = None
        self.mcr_st_init = None

        # Tryb wykresu
        self.current_plot_mode = 'SPECTRA'

        self.last_analysis_mode = None  # Pamięta ostatnią użytą analizę
        
        # Narzędzia wykresu
        self.zoom_rects = {}    # Przechowuje prostokąty
        self.zoom_start = {}    # Przechowuje punkt startu (x, y)
        self.initial_lims = {}
        self.pls_target_map = {}

        # Layout
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
        self.bind("<Button-1>", self._on_global_click)
        self.update_plot()

    def _create_control_widgets(self):
        self.control_frame.grid_rowconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(0, weight=1)
        
        # --- ZMIANA: dodano self. oraz command=self._on_tab_change ---
        self.tab_view = ctk.CTkTabview(self.control_frame, command=self._on_tab_change)
        self.tab_view.grid(row=0, column=0, sticky="nsew")

        data_tab = self.tab_view.add("Dane")
        preprocess_tab = self.tab_view.add("Preprocessing")
        analysis_tab = self.tab_view.add("Analiza")
        # -------------------------------------------------------------

        # Konfiguracja zawartości
        self._setup_data_tab(data_tab)
        self._setup_preprocess_tab(preprocess_tab)
        self._setup_analysis_tab(analysis_tab)

    def _on_tab_change(self):
        """Automatycznie zmienia widok w zależności od aktywnej zakładki."""
        tab_name = self.tab_view.get()

        if tab_name == "Dane":
            # Pokaż surowe dane (wyłącz podgląd preprocessingu)
            self.show_preprocessed_var.set(False)
            self.current_plot_mode = 'SPECTRA'
            
        elif tab_name == "Preprocessing":
            # Pokaż dane przetworzone (włącz podgląd)
            self.show_preprocessed_var.set(True)
            self.current_plot_mode = 'SPECTRA'
            
        elif tab_name == "Analiza":
            # Przywróć ostatnią analizę (jeśli była), w przeciwnym razie widma
            if self.last_analysis_mode:
                self.current_plot_mode = self.last_analysis_mode
            else:
                self.current_plot_mode = 'SPECTRA'

        self.update_plot()

    def _setup_data_tab(self, parent):
        frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        
        # Grid config
        row0 = ctk.CTkFrame(frame)
        row0.pack(fill="x", pady=5)
        ctk.CTkLabel(row0, text="N:").pack(side="left", padx=5)
        ctk.CTkEntry(row0, textvariable=self.n_var, width=50).pack(side="left")
        ctk.CTkLabel(row0, text="M:").pack(side="left", padx=5)
        ctk.CTkEntry(row0, textvariable=self.m_var, width=50).pack(side="left")
        ctk.CTkButton(row0, text="Utwórz Siatkę", command=self.safe_create_field_matrix).pack(side="left", padx=10)

        # Matrix container
        ctk.CTkLabel(frame, text="Macierz Pól").pack(anchor="w")
        self.field_matrix_frame = ctk.CTkFrame(frame)
        self.field_matrix_frame.pack(fill="x", pady=5)

        # Loading buttons
        row2 = ctk.CTkFrame(frame)
        row2.pack(fill="x", pady=5)
        ctk.CTkButton(row2, text="Wczytaj Pliki...", command=self._load_data_files).pack(side="left", fill="x", expand=True, padx=2)
        ctk.CTkButton(row2, text="Zaznacz Aktywne", command=self._select_all_active).pack(side="left", fill="x", expand=True, padx=2)
        ctk.CTkButton(row2, text="Odznacz", command=self._deselect_all).pack(side="left", fill="x", expand=True, padx=2)

        # Range
        row3 = ctk.CTkFrame(frame)
        row3.pack(fill="x", pady=5)
        ctk.CTkLabel(row3, text="Zakres (cm-1):").pack(side="left")
        ctk.CTkEntry(row3, textvariable=self.range_min_var, width=60).pack(side="left")
        ctk.CTkLabel(row3, text="-").pack(side="left")
        ctk.CTkEntry(row3, textvariable=self.range_max_var, width=60).pack(side="left")
        ctk.CTkButton(row3, text="Zastosuj", command=self._apply_wavenumber_range).pack(side="left", padx=5)
        ctk.CTkButton(row3, text="Reset", command=self._reset_wavenumber_range).pack(side="left", padx=5)

        # Project IO
        row4 = ctk.CTkFrame(frame)
        row4.pack(fill="x", pady=10)
        ctk.CTkButton(row4, text="Zapisz Projekt", command=self._save_project).pack(fill="x", pady=2)
        ctk.CTkButton(row4, text="Wczytaj Projekt", command=self._load_project).pack(fill="x", pady=2)
        ctk.CTkButton(row4, text="Eksportuj do Excela", command=self._export_to_xlsx).pack(fill="x", pady=2)

    def _setup_visual_tab(self, parent):
        frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        ctk.CTkSwitch(frame, text="Pokaż dane przetworzone", variable=self.show_preprocessed_var, command=self.update_plot).pack(pady=10)
        ctk.CTkButton(frame, text="Pokaż Heatmapę", command=self._run_heatmap).pack(fill="x", pady=5)

    def _setup_preprocess_tab(self, parent):
        # Główny kontener
        main_frame = ctk.CTkFrame(parent, fg_color="transparent")
        main_frame.pack(fill="both", expand=True)

        # GÓRA: Panel przycisków (Scrollowalny, żeby się zmieściło)
        buttons_frame = ctk.CTkScrollableFrame(main_frame, height=400) # Ograniczamy wysokość, żeby zostało miejsce na listę
        buttons_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        # Przełącznik widoku
        ctk.CTkSwitch(buttons_frame, text="Pokaż dane przetworzone (Podgląd)", 
                      variable=self.show_preprocessed_var, 
                      command=self.update_plot).pack(pady=(5, 5), anchor="w")
        
        # --- Sekcje Przycisków (jak wcześniej) ---
        # 1. SG SMOOTHING
        smooth_grp = ctk.CTkFrame(buttons_frame)
        smooth_grp.pack(fill="x", pady=5)
        ctk.CTkLabel(smooth_grp, text="Smoothing (Savitzky-Golay)", font=("Arial", 12, "bold")).pack()
        row_smooth = ctk.CTkFrame(smooth_grp, fg_color="transparent")
        row_smooth.pack(fill="x", pady=2)
        ctk.CTkEntry(row_smooth, textvariable=self.sg_smooth_window_var, placeholder_text="Win", width=40).pack(side="left", padx=2)
        ctk.CTkEntry(row_smooth, textvariable=self.sg_smooth_poly_var, placeholder_text="Poly", width=40).pack(side="left", padx=2)
        ctk.CTkButton(row_smooth, text="Dodaj", command=self._apply_sg_smoothing, width=60).pack(side="left", fill="x", expand=True, padx=2)

        # 2. SG 2ND DERIV
        sg_grp = ctk.CTkFrame(buttons_frame)
        sg_grp.pack(fill="x", pady=5)
        ctk.CTkLabel(sg_grp, text="2nd Deriv. (Savitzky-Golay)", font=("Arial", 12, "bold")).pack()
        row_sg = ctk.CTkFrame(sg_grp, fg_color="transparent")
        row_sg.pack(fill="x", pady=2)
        ctk.CTkEntry(row_sg, textvariable=self.sg_window_var, placeholder_text="Win", width=40).pack(side="left", padx=2)
        ctk.CTkEntry(row_sg, textvariable=self.sg_poly_var, placeholder_text="Poly", width=40).pack(side="left", padx=2)
        ctk.CTkButton(row_sg, text="Dodaj", command=self._apply_sg_filter, width=60).pack(side="left", fill="x", expand=True, padx=2)

        # 3. Normalizacja
        norm_grp = ctk.CTkFrame(buttons_frame)
        norm_grp.pack(fill="x", pady=5)
        ctk.CTkLabel(norm_grp, text="Normalizacja", font=("Arial", 12, "bold")).pack()
        grid_norm = ctk.CTkFrame(norm_grp, fg_color="transparent")
        grid_norm.pack(fill="x")
        ctk.CTkButton(grid_norm, text="SNV", command=self._apply_snv, width=60).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ctk.CTkButton(grid_norm, text="Min-Max", command=self._apply_min_max, width=60).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ctk.CTkButton(grid_norm, text="L1", command=self._apply_l1_norm, width=60).grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        ctk.CTkButton(grid_norm, text="MSC", command=self._apply_msc, width=60).grid(row=0, column=3, padx=2, pady=2, sticky="ew")
        for i in range(4): grid_norm.grid_columnconfigure(i, weight=1)

        # 4. ALS
        als_grp = ctk.CTkFrame(buttons_frame)
        als_grp.pack(fill="x", pady=5)
        ctk.CTkLabel(als_grp, text="ALS Baseline", font=("Arial", 12, "bold")).pack()
        row_als = ctk.CTkFrame(als_grp, fg_color="transparent")
        row_als.pack(fill="x")
        ctk.CTkEntry(row_als, textvariable=self.als_lambda_var, width=50).pack(side="left", padx=2)
        ctk.CTkEntry(row_als, textvariable=self.als_p_var, width=30).pack(side="left", padx=2)
        ctk.CTkButton(row_als, text="Dodaj", command=self._apply_als, width=60).pack(side="left", fill="x", expand=True, padx=2)

        # DÓŁ: Lista Kroków (Pipeline)
        ctk.CTkLabel(main_frame, text="Aktywne Kroki (Kolejność):", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        
        # Ramka z listą
        self.pipeline_scroll = ctk.CTkScrollableFrame(main_frame, height=200, fg_color="gray90" if ctk.get_appearance_mode()=="Light" else "gray20")
        self.pipeline_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Pusty stan
        ctk.CTkLabel(self.pipeline_scroll, text="(Brak kroków - dane surowe)", text_color="gray50").pack(pady=20)

    def _setup_analysis_tab(self, parent):
            frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
            frame.pack(fill="both", expand=True)

            # 1. ANALIZA CZYNNIKOWA (FA / SPEXFA)
            fa_frame = ctk.CTkFrame(frame)
            fa_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(fa_frame, text="Analiza Faktorowa (Malinowski / SPEXFA)", font=("Arial", 12, "bold")).pack()
            
            # PFA (Rank Analysis)
            ctk.CTkButton(fa_frame, text="Analiza Rangi (RE/IND)", command=self._run_fa_rank_analysis).pack(fill="x", pady=2)
            
            # SPEXFA
            row_spex = ctk.CTkFrame(fa_frame, fg_color="transparent")
            row_spex.pack(fill="x", pady=2)
            ctk.CTkEntry(row_spex, textvariable=self.spexfa_n_components_var, width=40).pack(side="left")
            ctk.CTkButton(row_spex, text="Uruchom SPEXFA", command=self._run_spexfa).pack(side="left", fill="x", expand=True)

            # Rekonstrukcja FA
            row_fa_rec = ctk.CTkFrame(fa_frame, fg_color="transparent")
            row_fa_rec.pack(fill="x", pady=2)
            ctk.CTkEntry(row_fa_rec, textvariable=self.fa_recon_components_var, width=40).pack(side="left")
            ctk.CTkButton(row_fa_rec, text="Rekonstrukcja FA", command=self._run_fa_reconstruction).pack(side="left", fill="x", expand=True)

            # 2. PCA
            pca_frame = ctk.CTkFrame(frame)
            pca_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(pca_frame, text="PCA (Główne Składowe)", font=("Arial", 12, "bold")).pack()
            
            row_pca = ctk.CTkFrame(pca_frame, fg_color="transparent")
            row_pca.pack(fill="x", pady=2)
            ctk.CTkEntry(row_pca, textvariable=self.pca_n_components_var, width=40).pack(side="left")
            ctk.CTkButton(row_pca, text="Oblicz PCA", command=self._run_pca).pack(side="left", fill="x", expand=True)
            
            row_pca_rec = ctk.CTkFrame(pca_frame, fg_color="transparent")
            row_pca_rec.pack(fill="x", pady=2)
            ctk.CTkEntry(row_pca_rec, textvariable=self.pca_recon_components_var, width=40).pack(side="left")
            ctk.CTkButton(row_pca_rec, text="Rekonstrukcja PCA", command=self._run_pca_reconstruction).pack(side="left", fill="x", expand=True)

            # 3. MCR-ALS
            mcr_frame = ctk.CTkFrame(frame)
            mcr_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(mcr_frame, text="MCR-ALS", font=("Arial", 12, "bold")).pack()
            
            row_mcr1 = ctk.CTkFrame(mcr_frame, fg_color="transparent")
            row_mcr1.pack(fill="x", pady=2)
            ctk.CTkEntry(row_mcr1, textvariable=self.mcr_n_components_var, width=40).pack(side="left")
            ctk.CTkCheckBox(row_mcr1, text="NNLS", variable=self.mcr_non_negative_var).pack(side="left", padx=5)
            ctk.CTkButton(row_mcr1, text="Start MCR", command=self._run_mcr_als).pack(side="left", fill="x", expand=True)

            row_mcr2 = ctk.CTkFrame(mcr_frame, fg_color="transparent")
            row_mcr2.pack(fill="x", pady=2)
            ctk.CTkButton(row_mcr2, text="Znane Widma...", command=self._load_mcr_st_init).pack(side="left", fill="x", expand=True)
            ctk.CTkEntry(row_mcr2, textvariable=self.mcr_st_fix_var, placeholder_text="Fix (np. 0,2)").pack(side="left", fill="x", expand=True)

            # 4. TENSOR (PARAFAC / Tucker)
            tens_frame = ctk.CTkFrame(frame)
            tens_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(tens_frame, text="Tensor (PARAFAC / Tucker)", font=("Arial", 12, "bold")).pack()
            
            # PARAFAC
            row_par = ctk.CTkFrame(tens_frame, fg_color="transparent")
            row_par.pack(fill="x", pady=2)
            ctk.CTkEntry(row_par, textvariable=self.parafac_rank_var, width=40).pack(side="left")
            ctk.CTkButton(row_par, text="PARAFAC", command=self._run_parafac).pack(side="left", fill="x", expand=True)

            # Tucker
            ctk.CTkLabel(tens_frame, text="Rangi Tucker (W, N, M):", font=("Arial", 10)).pack(anchor="w")
            row_tuck = ctk.CTkFrame(tens_frame, fg_color="transparent")
            row_tuck.pack(fill="x", pady=2)
            ctk.CTkEntry(row_tuck, textvariable=self.tucker_rank_w_var, width=30).pack(side="left")
            ctk.CTkEntry(row_tuck, textvariable=self.tucker_rank_n_var, width=30).pack(side="left")
            ctk.CTkEntry(row_tuck, textvariable=self.tucker_rank_m_var, width=30).pack(side="left")
            ctk.CTkButton(row_tuck, text="Tucker", command=self._run_tucker).pack(side="left", fill="x", expand=True)

            # Rekonstrukcja Tensora
            row_tens_rec = ctk.CTkFrame(tens_frame, fg_color="transparent")
            row_tens_rec.pack(fill="x", pady=5)
            ctk.CTkLabel(row_tens_rec, text="Rekonstrukcja: ").pack(side="left")
            ctk.CTkEntry(row_tens_rec, textvariable=self.tensor_recon_components_var, width=30).pack(side="left")
            ctk.CTkButton(row_tens_rec, text="z PARAFAC", command=lambda: self._run_tensor_recon('parafac')).pack(side="left", padx=2)
            ctk.CTkButton(row_tens_rec, text="z MCR", command=lambda: self._run_tensor_recon('mcr')).pack(side="left", padx=2)

            # 5. 2D-COS
            cos_frame = ctk.CTkFrame(frame)
            cos_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(cos_frame, text="2D-COS", font=("Arial", 12, "bold")).pack()
            ctk.CTkOptionMenu(cos_frame, variable=self.cos_axis_var, values=["Analizuj Wiersze (N)", "Analizuj Kolumny (M)"]).pack(fill="x", pady=2)
            ctk.CTkButton(cos_frame, text="Uruchom 3D-COS", command=self._run_3dcos).pack(fill="x")
            
            # Ukryty slider (pokazywany w update_plot)
            self.cos_slider_frame = ctk.CTkFrame(cos_frame, fg_color="transparent")
            self.cos_slider_frame.pack(fill="x")
            ctk.CTkLabel(self.cos_slider_frame, text="Plaster:").pack(side="left")
            self.cos_slider = ctk.CTkSlider(self.cos_slider_frame, from_=0, to=1, number_of_steps=1, command=self._on_cos_slider)
            self.cos_slider.pack(side="left", fill="x", expand=True)
            self.cos_slider_frame.pack_forget()

            # 6. MANIFOLD
            man_frame = ctk.CTkFrame(frame)
            man_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(man_frame, text="Redukcja Wymiaru", font=("Arial", 12, "bold")).pack()
            row_man = ctk.CTkFrame(man_frame, fg_color="transparent")
            row_man.pack(fill="x", pady=2)
            ctk.CTkButton(row_man, text="UMAP", command=self._run_umap).pack(side="left", fill="x", expand=True)
            ctk.CTkButton(row_man, text="t-SNE", command=self._run_tsne).pack(side="left", fill="x", expand=True)

    # --- ZARZĄDZANIE DANYMI ---
    def safe_create_field_matrix(self):
        if self.original_tensor_data is not None:
            if not messagebox.askyesno("Potwierdzenie", "Utworzenie nowej siatki usunie dane. Kontynuować?"):
                return
        self._create_field_matrix()

    def _create_field_matrix(self):
        for widget in self.field_matrix_frame.winfo_children(): widget.destroy()
        self.field_matrix_widgets.clear()
        self.field_matrix_status.clear()
        self.selected_coords.clear()
        self.original_tensor_data = None
        self.preprocessed_tensor = None
        self._reset_analysis_results()

        try:
            self.n_rows = int(self.n_var.get())
            self.m_cols = int(self.m_var.get())
        except ValueError:
            return

        for c_idx in range(self.m_cols): self.field_matrix_frame.grid_columnconfigure(c_idx, weight=1)
        for r_idx in range(self.n_rows):
            for c_idx in range(self.m_cols):
                coords = (r_idx, c_idx)
                cell_text = f"({r_idx},{c_idx})\nPusty"
                btn = ctk.CTkButton(self.field_matrix_frame, text=cell_text, 
                                    height=60, fg_color=STATUS_COLORS['EMPTY'])
                btn.grid(row=r_idx, column=c_idx, padx=1, pady=1, sticky="nsew")
                
                # --- NAPRAWA PPM DLA MAC OS ---
                # Windows/Linux
                btn.bind("<Button-3>", lambda e, c=coords: self._on_cell_right_click(e, c))
                # macOS (często Button-2 to prawy przycisk)
                btn.bind("<Button-2>", lambda e, c=coords: self._on_cell_right_click(e, c))
                # macOS (Control + Kliknięcie = Prawy Przycisk)
                btn.bind("<Control-Button-1>", lambda e, c=coords: self._on_cell_right_click(e, c))
                # ------------------------------

                btn.configure(command=lambda c=coords: self._on_cell_left_click(c))
                self.field_matrix_widgets[coords] = btn
                self.field_matrix_status[coords] = 'EMPTY'

    def _mark_cell_as_missing(self, coords):
        self._update_cell_status(coords, 'MISSING', f"{coords}\nBRAK")
        if self.original_tensor_data is not None:
            self.original_tensor_data[:, coords[0], coords[1]] = np.nan
        # Reset range ensures tensor_data is updated
        self._reset_wavenumber_range()

    def _mark_cell_as_empty(self, coords):
        self._update_cell_status(coords, 'EMPTY', f"{coords}\nPusty")
        if self.original_tensor_data is not None:
            self.original_tensor_data[:, coords[0], coords[1]] = np.nan
        self._reset_wavenumber_range()

# --- METODY WYPEŁNIANIA DANYCH (Grid Operations) ---

    def _fill_cell_with_zeros(self, coords):
        if self.original_tensor_data is None: return
        r, c = coords
        # Wstawiamy zera
        zeros = np.zeros(len(self.original_wavenumbers))
        self.original_tensor_data[:, r, c] = zeros
        
        # Aktualizujemy też dane robocze, jeśli istnieją
        if self.tensor_data is not None:
            # Uwaga: tensor_data może mieć inną długość (wavenumbers) niż original,
            # więc bezpieczniej jest zresetować zakres lub przeliczyć maskę.
            # Dla uproszczenia tutaj aktualizujemy oryginał i robimy reset zakresu (najbezpieczniej)
            pass 

        self._update_cell_status(coords, 'FILLED_ZERO', f"{coords}\nWypełniono (0)")
        self._reset_wavenumber_range() # Odświeża tensor_data
        self.update_plot()

    def _fill_cell_with_row_mean(self, coords):
        if self.original_tensor_data is None: return
        r, c = coords
        try:
            # Średnia z całego wiersza (ignorując NaN)
            mean_spectrum = np.nanmean(self.original_tensor_data[:, r, :], axis=1)
            
            if np.all(np.isnan(mean_spectrum)): 
                raise ValueError("Brak danych w tym wierszu.")
            
            self.original_tensor_data[:, r, c] = mean_spectrum
            self._update_cell_status(coords, 'FILLED_ROW_MEAN', f"{coords}\nWypełniono (Śr. Wiersza)")
            self._reset_wavenumber_range()
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można obliczyć średniej wiersza:\n{e}")

    def _fill_cell_with_col_mean(self, coords):
        if self.original_tensor_data is None: return
        r, c = coords
        try:
            # Średnia z całej kolumny
            mean_spectrum = np.nanmean(self.original_tensor_data[:, :, c], axis=1)
            
            if np.all(np.isnan(mean_spectrum)): 
                raise ValueError("Brak danych w tej kolumnie.")
            
            self.original_tensor_data[:, r, c] = mean_spectrum
            self._update_cell_status(coords, 'FILLED_COL_MEAN', f"{coords}\nWypełniono (Śr. Kolumny)")
            self._reset_wavenumber_range()
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można obliczyć średniej kolumny:\n{e}")

    # --- NOWE METODY: ŚREDNIA Z SĄSIADÓW ---

    def _fill_cell_with_row_neighbors(self, coords):
        """Wypełnia komórkę średnią z bezpośredniego lewego i prawego sąsiada."""
        if self.original_tensor_data is None: return
        r, c = coords
        
        neighbors = []
        # Lewy sąsiad
        if c > 0: 
            neighbors.append(self.original_tensor_data[:, r, c-1])
        # Prawy sąsiad
        if c < self.m_cols - 1:
            neighbors.append(self.original_tensor_data[:, r, c+1])
            
        self._calculate_neighbors_mean(coords, neighbors, "Sąsiedzi (Wiersz)")

    def _fill_cell_with_col_neighbors(self, coords):
        """Wypełnia komórkę średnią z bezpośredniego górnego i dolnego sąsiada."""
        if self.original_tensor_data is None: return
        r, c = coords
        
        neighbors = []
        # Górny sąsiad
        if r > 0:
            neighbors.append(self.original_tensor_data[:, r-1, c])
        # Dolny sąsiad
        if r < self.n_rows - 1:
            neighbors.append(self.original_tensor_data[:, r+1, c])
            
        self._calculate_neighbors_mean(coords, neighbors, "Sąsiedzi (Kolumna)")

    def _calculate_neighbors_mean(self, coords, neighbors, label):
        """Funkcja pomocnicza do liczenia średniej z listy widm."""
        try:
            # Filtrujemy tylko te widma, które nie są samym NaN
            valid_neighbors = [n for n in neighbors if not np.all(np.isnan(n))]
            
            if not valid_neighbors:
                raise ValueError("Brak ważnych sąsiadów (wszyscy są puści lub NaN).")
            
            # Obliczamy średnią
            mean_spec = np.nanmean(np.array(valid_neighbors), axis=0)
            
            # Wstawiamy do danych
            self.original_tensor_data[:, coords[0], coords[1]] = mean_spec
            
            self._update_cell_status(coords, 'FILLED_NEIGHBOR', f"{coords}\n{label}")
            self._reset_wavenumber_range()
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Błąd Interpolacji", f"Nie można uzupełnić z sąsiadów:\n{e}")    

    def _load_data_files(self):
        cells = [c for c, s in self.field_matrix_status.items() if s == 'EMPTY']
        if not cells: return
        
        paths = filedialog.askopenfilenames(title=f"Wybierz {len(cells)} plików CSV")
        if len(paths) != len(cells):
            messagebox.showerror("Błąd", f"Wybrano {len(paths)} plików, oczekiwano {len(cells)}")
            return

        file_iter = iter(paths)
        is_first = (self.original_wavenumbers is None)
        temp_data = None

        for coords in cells:
            path = next(file_iter)
            try:
                wav, absorb = data_io.load_single_csv(path)
                
                if is_first:
                    self.original_wavenumbers = wav
                    temp_data = np.full((len(wav), self.n_rows, self.m_cols), np.nan)
                    is_first = False
                
                if np.array_equal(self.original_wavenumbers, wav):
                    temp_data[:, coords[0], coords[1]] = absorb
                    self._update_cell_status(coords, 'LOADED', f"{coords}\nOK")
                else:
                    self._update_cell_status(coords, 'ERROR', "BŁĄD X")
            except Exception:
                self._update_cell_status(coords, 'ERROR', "BŁĄD PLIKU")

        if self.original_tensor_data is None:
            self.original_tensor_data = temp_data
        else:
            # Merge logic if needed (simplified here)
            pass

        self._reset_wavenumber_range()
        self.update_plot()

    def _update_cell_status(self, coords, status, text):
        self.field_matrix_status[coords] = status
        btn = self.field_matrix_widgets[coords]
        btn.configure(text=text, fg_color=STATUS_COLORS.get(status.split('_')[0], 'gray'))

    def _on_cell_left_click(self, coords):
        status = self.field_matrix_status.get(coords)
        
        # --- POPRAWKA: Sprawdzamy, czy status zaczyna się od 'FILLED' ---
        # Wcześniej było: if status not in ['LOADED', 'FILLED']: return
        if status != 'LOADED' and not (status and status.startswith('FILLED')): 
            return
        # ---------------------------------------------------------------

        btn = self.field_matrix_widgets[coords]
        
        if coords in self.selected_coords:
            self.selected_coords.remove(coords)
            btn.configure(border_width=0)
        else:
            self.selected_coords.add(coords)
            btn.configure(border_width=3, border_color=SELECTED_BORDER_COLOR)
        
        self.current_plot_mode = 'SPECTRA'
        self.update_plot()

    def _on_cell_right_click(self, event, coords):
        menu = Menu(self, tearoff=0)
        status = self.field_matrix_status.get(coords)
        
        # Opcje dostępne zawsze (poza poprawnie załadowanymi)
        if status != 'LOADED':
            menu.add_command(label="Oznacz jako BRAK DANYCH", command=lambda: self._mark_cell_as_missing(coords))
            menu.add_command(label="Oznacz jako Pusty", command=lambda: self._mark_cell_as_empty(coords))
        
        # Opcje wypełniania (dostępne jeśli mamy jakieś dane w tensorze)
        if self.original_tensor_data is not None:
            menu.add_separator()
            # Podmenu "Wypełnij..."
            fill_menu = Menu(menu, tearoff=0)
            menu.add_cascade(label="Wypełnij danymi...", menu=fill_menu)
            
            fill_menu.add_command(label="Zerami (0.0)", command=lambda: self._fill_cell_with_zeros(coords))
            fill_menu.add_separator()
            fill_menu.add_command(label="Średnia z WIERSZA (całego)", command=lambda: self._fill_cell_with_row_mean(coords))
            fill_menu.add_command(label="Średnia z KOLUMNY (całej)", command=lambda: self._fill_cell_with_col_mean(coords))
            fill_menu.add_separator()
            # --- NOWE OPCJE ---
            fill_menu.add_command(label="Średnia z SĄSIADÓW (Lewy/Prawy)", command=lambda: self._fill_cell_with_row_neighbors(coords))
            fill_menu.add_command(label="Średnia z SĄSIADÓW (Góra/Dół)", command=lambda: self._fill_cell_with_col_neighbors(coords))

        # Opcje resetu
        if status.startswith('LOADED') or status.startswith('FILLED'):
            menu.add_separator()
            menu.add_command(label="Resetuj (Oznacz jako Pusty)", command=lambda: self._mark_cell_as_empty(coords))
            
        menu.post(event.x_root, event.y_root)

    def _select_all_active(self):
        for coords, status in self.field_matrix_status.items():
            # --- POPRAWKA: ---
            is_filled = status and status.startswith('FILLED')
            if (status == 'LOADED' or is_filled) and coords not in self.selected_coords:
            # -----------------
                self.selected_coords.add(coords)
                self.field_matrix_widgets.get(coords).configure(border_width=3, border_color=SELECTED_BORDER_COLOR)
        self.update_plot()

    def _deselect_all(self):
        for coords in self.selected_coords:
            self.field_matrix_widgets[coords].configure(border_width=0)
        self.selected_coords.clear()
        self.update_plot()

    def _apply_wavenumber_range(self):
        if self.original_tensor_data is None: return
        try:
            mn, mx = float(self.range_min_var.get()), float(self.range_max_var.get())
            mask = np.where((self.original_wavenumbers >= mn) & (self.original_wavenumbers <= mx))[0]
            if len(mask) == 0: raise ValueError
            self.wavenumbers = self.original_wavenumbers[mask]
            self.tensor_data = self.original_tensor_data[mask, :, :]
            self.preprocessed_tensor = None
            self.update_plot()
        except ValueError:
            messagebox.showerror("Błąd", "Niepoprawny zakres")

    def _reset_wavenumber_range(self):
        if self.original_tensor_data is not None:
            self.wavenumbers = np.copy(self.original_wavenumbers)
            self.tensor_data = np.copy(self.original_tensor_data)
            self.range_min_var.set(f"{np.min(self.wavenumbers):.2f}")
            self.range_max_var.set(f"{np.max(self.wavenumbers):.2f}")
            self.update_plot()

    # --- PREPROCESSING ---
    def _get_active_data(self):
        # Nowa logika: decyduje przełącznik "Pokaż dane przetworzone" (show_preprocessed_var)
        # Jeśli jest włączony I mamy przeliczone dane -> zwracamy je.
        # W przeciwnym razie -> zwracamy dane surowe.
        if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None:
            return self.preprocessed_tensor
        return self.tensor_data

    def _process_data(self, func, name):
        data = self._get_active_data()
        if data is None: return
        try:
            # Funkcje z modułu zwracają NOWY tensor
            new_tensor = func(data)
            self.preprocessed_tensor = new_tensor
            self.show_preprocessed_var.set(True)
            self.update_plot()
            messagebox.showinfo("Sukces", f"Zastosowano: {name}")
        except Exception as e:
            messagebox.showerror("Błąd", f"{e}")

 # --- ZAKTUALIZOWANE WRAPPERY ---

    def _apply_sg_smoothing(self):
        try:
            w = int(self.sg_smooth_window_var.get())
            p = int(self.sg_smooth_poly_var.get())
            label = f"Smoothing (w={w}, p={p})"
            self._add_pipeline_step('SG', {'w': w, 'poly': p, 'deriv': 0}, label)
        except ValueError: messagebox.showerror("Błąd", "Liczby całkowite wymagane")

    def _apply_sg_filter(self):
        try:
            w = int(self.sg_window_var.get())
            p = int(self.sg_poly_var.get())
            label = f"2nd Deriv (w={w}, p={p})"
            self._add_pipeline_step('SG', {'w': w, 'poly': p, 'deriv': 2}, label)
        except ValueError: messagebox.showerror("Błąd", "Liczby całkowite wymagane")

    def _apply_snv(self):     self._add_pipeline_step('SNV', label="SNV")
    def _apply_min_max(self): self._add_pipeline_step('MinMax', label="Min-Max")
    def _apply_l1_norm(self): self._add_pipeline_step('L1', label="Norma L1")
    def _apply_msc(self):     self._add_pipeline_step('MSC', label="MSC")

    def _apply_als(self):
        try:
            lam = float(self.als_lambda_var.get())
            p = float(self.als_p_var.get())
            label = f"ALS (λ={lam:.0e}, p={p})"
            self._add_pipeline_step('ALS', {'lam': lam, 'p': p}, label)
        except ValueError: messagebox.showerror("Błąd", "Liczby wymagane")

    # --- NOWY SILNIK PIPELINE (Dodaj do klasy ChemTensorApp) ---

    def _add_pipeline_step(self, method_type, params=None, label=None):
        """Dodaje krok do listy i odświeża wynik."""
        step = {
            'type': method_type,
            'params': params if params else {},
            'label': label if label else method_type
        }
        self.pipeline_steps.append(step)
        self._refresh_pipeline_ui()
        self._run_pipeline()

    def _remove_pipeline_step(self, index):
        """Usuwa krok o danym indeksie."""
        if 0 <= index < len(self.pipeline_steps):
            del self.pipeline_steps[index]
            self._refresh_pipeline_ui()
            self._run_pipeline()

    def _move_pipeline_step(self, index, direction):
        """Przesuwa krok w górę (-1) lub w dół (+1)."""
        new_index = index + direction
        if 0 <= new_index < len(self.pipeline_steps):
            self.pipeline_steps[index], self.pipeline_steps[new_index] = \
                self.pipeline_steps[new_index], self.pipeline_steps[index]
            self._refresh_pipeline_ui()
            self._run_pipeline()

    def _run_pipeline(self):
        """Bierze surowe dane i przepuszcza przez wszystkie kroki."""
        if self.tensor_data is None: return

        # Zawsze startujemy od czystych danych (kopiujemy, żeby nie psuć oryginału)
        current_data = np.copy(self.tensor_data)
        
        try:
            for step in self.pipeline_steps:
                m_type = step['type']
                p = step['params']
                
                if m_type == 'SNV':
                    current_data = prep.apply_snv(current_data)
                elif m_type == 'MinMax':
                    current_data = prep.apply_min_max(current_data)
                elif m_type == 'L1':
                    current_data = prep.apply_l1_norm(current_data)
                elif m_type == 'MSC':
                    current_data = prep.apply_msc(current_data)
                elif m_type == 'ALS':
                    current_data = prep.correction_als(current_data, p['lam'], p['p'])
                elif m_type == 'SG':
                    current_data = prep.apply_savgol(current_data, p['w'], p['poly'], p['deriv'])
            
            self.preprocessed_tensor = current_data
            
            # Automatycznie włącz podgląd, jeśli dodano kroki
            if self.pipeline_steps:
                self.show_preprocessed_var.set(True)
            
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Błąd Pipeline", f"Błąd w kroku {m_type}:\n{e}")
            # W razie błędu resetujemy do surowych, żeby nie pokazywać głupot
            self.preprocessed_tensor = None 

    def _refresh_pipeline_ui(self):
        """Rysuje listę kafelków w GUI."""
        # Czyścimy kontener
        for widget in self.pipeline_scroll.winfo_children():
            widget.destroy()

        for i, step in enumerate(self.pipeline_steps):
            # Ramka dla jednego kroku
            card = ctk.CTkFrame(self.pipeline_scroll, fg_color="gray85" if ctk.get_appearance_mode()=="Light" else "gray25")
            card.pack(fill="x", pady=2, padx=2)
            
            # Etykieta (np. "1. SG (11, 2)")
            lbl_text = f"{i+1}. {step['label']}"
            ctk.CTkLabel(card, text=lbl_text, font=("Arial", 11)).pack(side="left", padx=5)
            
            # Przyciski akcji (po prawej)
            ctk.CTkButton(card, text="X", width=25, height=25, fg_color="#C0392B", 
                          command=lambda idx=i: self._remove_pipeline_step(idx)).pack(side="right", padx=2)
            
            if i < len(self.pipeline_steps) - 1: # Przycisk Dół
                ctk.CTkButton(card, text="▼", width=25, height=25, fg_color="gray50",
                              command=lambda idx=i: self._move_pipeline_step(idx, 1)).pack(side="right", padx=1)
            
            if i > 0: # Przycisk Góra
                ctk.CTkButton(card, text="▲", width=25, height=25, fg_color="gray50",
                              command=lambda idx=i: self._move_pipeline_step(idx, -1)).pack(side="right", padx=1)
    # --- ANALIZA ---
    def _get_selection_matrix(self):
        if not self.selected_coords:
            messagebox.showerror("Błąd", "Brak zaznaczenia")
            return None, None
        
        data_source = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
        matrix = []
        labels = []
        for coords in sorted(list(self.selected_coords)):
            r, c = coords
            spec = data_source[:, r, c]
            if not np.isnan(spec).any():
                matrix.append(spec)
                labels.append(str(coords))
        return np.array(matrix), labels

    def _run_pca(self):
        X, labels = self._get_selection_matrix()
        if X is None: return
        try:
            n = int(self.pca_n_components_var.get())
            self.pca_results = anal.run_pca(X, n)
            self.pca_results['labels'] = labels
            self.current_plot_mode = 'PCA'
            self.last_analysis_mode = 'PCA'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd PCA", f"{e}")

# ZAKTUALIZUJ ISTNIEJĄCĄ METODĘ W main.py
    def _run_mcr_als(self):
        data = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
        if data is None: return
        try:
            n = int(self.mcr_n_components_var.get())
            
            # Parsowanie indeksów fix
            st_fix_indices = []
            if self.mcr_st_fix_var.get():
                st_fix_indices = [int(i) for i in self.mcr_st_fix_var.get().split(',')]

            self.mcr_results = anal.run_mcr_als(
                data, n, 
                self.mcr_non_negative_var.get(), 
                self.mcr_norm_var.get(),
                st_init=self.mcr_st_init,
                st_fix_indices=st_fix_indices
            )
            self.current_plot_mode = 'MCR'
            self.last_analysis_mode = 'MCR'
            self._update_pls_target_options() # Aktualizujemy listę dla PLS!
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd MCR", f"{e}")

# ZAKTUALIZUJ ISTNIEJĄCĄ METODĘ W main.py
    def _run_parafac(self):
        # ... (pobieranie danych - bez zmian) ...
        data = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
        if data is None: return
        try:
            rank = int(self.parafac_rank_var.get())
            self.parafac_results = anal.run_parafac(data, rank, self.parafac_non_negative_var.get())
            self.current_plot_mode = 'PARAFAC'
            self.last_analysis_mode = 'PARAFAC'
            self._update_pls_target_options() # <--- DODAJ TO
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd PARAFAC", f"{e}")

    def _run_umap(self):
        X, labels = self._get_selection_matrix()
        if X is None: return
        try:
            emb = anal.run_umap(X)
            self.manifold_results = {'scores': emb, 'labels': labels, 'method_name': 'UMAP'}
            self.current_plot_mode = 'MANIFOLD'
            self.last_analysis_mode = 'MANIFOLD'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd UMAP", f"{e}")

    def _run_tsne(self):
        X, labels = self._get_selection_matrix()
        if X is None: return
        try:
            emb = anal.run_tsne(X)
            self.manifold_results = {'scores': emb, 'labels': labels, 'method_name': 't-SNE'}
            self.current_plot_mode = 'MANIFOLD'
            self.last_analysis_mode = 'MANIFOLD'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd t-SNE", f"{e}")

    def _run_heatmap(self):
        self.current_plot_mode = 'HEATMAP'
        self.last_analysis_mode = 'HEATMAP'
        self.update_plot()
    
    # --- DODATKI: PLS, 2D-COS, MCR-ADVANCED (Wklej do main.py w klasie ChemTensorApp) ---

    def _update_pls_target_options(self):
        """Aktualizuje listę dostępnych celów (Y) dla PLS na podstawie wyników MCR/PARAFAC."""
        self.pls_target_map.clear()
        options = []

        # Pobieranie stężeń z MCR
        if self.mcr_results:
            k = self.mcr_results['rank']
            for i in range(k):
                name = f"MCR Stężenie Skł. {i + 1}"
                self.pls_target_map[name] = self.mcr_results['C'][:, i]
                options.append(name)

        # Pobieranie trendów z PARAFAC
        if self.parafac_results:
            k = self.parafac_results['factors'][0].shape[1]
            factors = self.parafac_results['factors']
            for i in range(k):
                name_n = f"PARAFAC Trend N Skł. {i + 1}"
                # Powtarzamy wartości dla każdego m, żeby pasowało do spłaszczonego tensora
                trend_n = np.repeat(factors[1][:, i], self.m_cols)
                self.pls_target_map[name_n] = trend_n
                options.append(name_n)
                
                name_m = f"PARAFAC Trend M Skł. {i + 1}"
                # Kafelkujemy wartości dla każdego n
                trend_m = np.tile(factors[2][:, i], self.n_rows)
                self.pls_target_map[name_m] = trend_m
                options.append(name_m)

        if not options:
            options = ["Brak wyników (uruchom MCR/PARAFAC)"]
        
        # Aktualizacja Menu (jeśli istnieje)
        if hasattr(self, 'pls_target_menu'):
            self.pls_target_menu.configure(values=options)
            self.pls_target_var.set(options[0])

    def _run_pls(self):
        target_name = self.pls_target_var.get()
        if target_name not in self.pls_target_map:
            messagebox.showerror("Błąd", "Wybierz poprawny cel z listy.")
            return
        
        y = self.pls_target_map[target_name]
        data = self._get_active_data()
        if data is None: return

        try:
            self.pls_results = anal.run_pls(data, y)
            self.current_plot_mode = 'PLS_RESULTS'
            self.last_analysis_mode = 'PLS_RESULTS'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd PLS", f"{e}")

    def _run_3dcos(self):
        data = self._get_active_data()
        if data is None: return
        
        try:
            tensor_no_nan = np.nan_to_num(data, nan=0.0)
            w, n, m = tensor_no_nan.shape
            axis = self.cos_axis_var.get()
            
            phi_list, psi_list = [], []

            if "Wiersze" in axis: # N
                modulator_size = n
                if m < 2: raise ValueError("Za mało kolumn do korelacji wierszy.")
                for i in range(n):
                    phi, psi = anal.calculate_2dcos(tensor_no_nan[:, i, :])
                    phi_list.append(phi); psi_list.append(psi)
            else: # M
                modulator_size = m
                if n < 2: raise ValueError("Za mało wierszy do korelacji kolumn.")
                for j in range(m):
                    phi, psi = anal.calculate_2dcos(tensor_no_nan[:, :, j])
                    phi_list.append(phi); psi_list.append(psi)
            
            self.cos_3d_results = {
                'phi': np.stack(phi_list, axis=-1),
                'psi': np.stack(psi_list, axis=-1)
            }
            
            # Konfiguracja slidera
            if hasattr(self, 'cos_slider'):
                self.cos_slider.configure(to=modulator_size - 1, number_of_steps=modulator_size - 1)
                self.cos_slider.set(0)
                self.cos_slice_var.set(0)
            
            self.current_plot_mode = '3DCOS_SLICER'
            self.last_analysis_mode = '3DCOS_SLICER'
            self.update_plot()
            
        except Exception as e: messagebox.showerror("Błąd 2D-COS", f"{e}")

    def _load_mcr_st_init(self):
        if self.wavenumbers is None: 
            messagebox.showerror("Błąd", "Brak osi liczb falowych.")
            return
            
        paths = filedialog.askopenfilenames(title="Wybierz pliki ze znanymi widmami")
        if not paths: return
        
        loaded_spectra = []
        for p in paths:
            try:
                wav, absorb = data_io.load_single_csv(p)
                # Sprawdzenie zgodności osi X
                if not np.array_equal(self.wavenumbers, wav):
                    raise ValueError("Niezgodna oś liczb falowych")
                loaded_spectra.append(absorb)
            except Exception as e:
                messagebox.showerror("Błąd", f"Plik {p}: {e}")
                return
        
        self.mcr_st_init = np.array(loaded_spectra)
        indices_str = ",".join(map(str, range(len(paths))))
        self.mcr_st_fix_var.set(indices_str)
        messagebox.showinfo("Info", f"Wczytano {len(paths)} widm. Sugerowane indeksy: {indices_str}")

    def _on_cos_slider(self, val):
        self.cos_slice_var.set(int(val))
        self.update_plot()# --- DODATKI: PLS, 2D-COS, MCR-ADVANCED (Wklej do main.py w klasie ChemTensorApp) ---

    def _update_pls_target_options(self):
        """Aktualizuje listę dostępnych celów (Y) dla PLS na podstawie wyników MCR/PARAFAC."""
        self.pls_target_map.clear()
        options = []

        # Pobieranie stężeń z MCR
        if self.mcr_results:
            k = self.mcr_results['rank']
            for i in range(k):
                name = f"MCR Stężenie Skł. {i + 1}"
                self.pls_target_map[name] = self.mcr_results['C'][:, i]
                options.append(name)

        # Pobieranie trendów z PARAFAC
        if self.parafac_results:
            k = self.parafac_results['factors'][0].shape[1]
            factors = self.parafac_results['factors']
            for i in range(k):
                name_n = f"PARAFAC Trend N Skł. {i + 1}"
                # Powtarzamy wartości dla każdego m, żeby pasowało do spłaszczonego tensora
                trend_n = np.repeat(factors[1][:, i], self.m_cols)
                self.pls_target_map[name_n] = trend_n
                options.append(name_n)
                
                name_m = f"PARAFAC Trend M Skł. {i + 1}"
                # Kafelkujemy wartości dla każdego n
                trend_m = np.tile(factors[2][:, i], self.n_rows)
                self.pls_target_map[name_m] = trend_m
                options.append(name_m)

        if not options:
            options = ["Brak wyników (uruchom MCR/PARAFAC)"]
        
        # Aktualizacja Menu (jeśli istnieje)
        if hasattr(self, 'pls_target_menu'):
            self.pls_target_menu.configure(values=options)
            self.pls_target_var.set(options[0])

    def _run_pls(self):
        target_name = self.pls_target_var.get()
        if target_name not in self.pls_target_map:
            messagebox.showerror("Błąd", "Wybierz poprawny cel z listy.")
            return
        
        y = self.pls_target_map[target_name]
        data = self._get_active_data()
        if data is None: return

        try:
            self.pls_results = anal.run_pls(data, y)
            self.current_plot_mode = 'PLS_RESULTS'
            self.last_analysis_mode = 'PLS_RESULTS'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd PLS", f"{e}")

    def _run_3dcos(self):
        data = self._get_active_data()
        if data is None: return
        
        try:
            tensor_no_nan = np.nan_to_num(data, nan=0.0)
            w, n, m = tensor_no_nan.shape
            axis = self.cos_axis_var.get()
            
            phi_list, psi_list = [], []

            if "Wiersze" in axis: # N
                modulator_size = n
                if m < 2: raise ValueError("Za mało kolumn do korelacji wierszy.")
                for i in range(n):
                    phi, psi = anal.calculate_2dcos(tensor_no_nan[:, i, :])
                    phi_list.append(phi); psi_list.append(psi)
            else: # M
                modulator_size = m
                if n < 2: raise ValueError("Za mało wierszy do korelacji kolumn.")
                for j in range(m):
                    phi, psi = anal.calculate_2dcos(tensor_no_nan[:, :, j])
                    phi_list.append(phi); psi_list.append(psi)
            
            self.cos_3d_results = {
                'phi': np.stack(phi_list, axis=-1),
                'psi': np.stack(psi_list, axis=-1)
            }
            
            # Konfiguracja slidera
            if hasattr(self, 'cos_slider'):
                self.cos_slider.configure(to=modulator_size - 1, number_of_steps=modulator_size - 1)
                self.cos_slider.set(0)
                self.cos_slice_var.set(0)
            
            self.current_plot_mode = '3DCOS_SLICER'
            self.last_analysis_mode = '3DCOS_SLICER'
            self.update_plot()
            
        except Exception as e: messagebox.showerror("Błąd 2D-COS", f"{e}")

    def _load_mcr_st_init(self):
        if self.wavenumbers is None: 
            messagebox.showerror("Błąd", "Brak osi liczb falowych.")
            return
            
        paths = filedialog.askopenfilenames(title="Wybierz pliki ze znanymi widmami")
        if not paths: return
        
        loaded_spectra = []
        for p in paths:
            try:
                wav, absorb = data_io.load_single_csv(p)
                # Sprawdzenie zgodności osi X
                if not np.array_equal(self.wavenumbers, wav):
                    raise ValueError("Niezgodna oś liczb falowych")
                loaded_spectra.append(absorb)
            except Exception as e:
                messagebox.showerror("Błąd", f"Plik {p}: {e}")
                return
        
        self.mcr_st_init = np.array(loaded_spectra)
        indices_str = ",".join(map(str, range(len(paths))))
        self.mcr_st_fix_var.set(indices_str)
        messagebox.showinfo("Info", f"Wczytano {len(paths)} widm. Sugerowane indeksy: {indices_str}")

    def _on_cos_slider(self, val):
        self.cos_slice_var.set(int(val))
        self.update_plot()

# --- DODAJ TE METODY DO KLASY ChemTensorApp w main.py ---

    # 1. Rekonstrukcja PCA
    def _run_pca_reconstruction(self):
        if self.pca_results is None:
            messagebox.showerror("Błąd", "Najpierw uruchom PCA")
            return
        try:
            k = int(self.pca_recon_components_var.get())
            self.pca_results.update(anal.run_pca_reconstruction(self.pca_results, k))
            self.current_plot_mode = 'RECONSTRUCTION'
            self.last_analysis_mode = 'RECONSTRUCTION'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd", f"{e}")

    # 2. Tucker
    def _run_tucker(self):
        data = self._get_active_data()
        if data is None: return
        try:
            rw = int(self.tucker_rank_w_var.get())
            rn = int(self.tucker_rank_n_var.get())
            rm = int(self.tucker_rank_m_var.get())
            self.tucker_results = anal.run_tucker(data, [rw, rn, rm])
            self.current_plot_mode = 'TUCKER_RESULTS'
            self.last_analysis_mode = 'TUCKER_RESULTS'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd Tucker", f"{e}")

    # 3. Analiza Faktorowa (Rank Analysis)
    def _run_fa_rank_analysis(self):
        X, labels = self._get_selection_matrix()
        if X is None: return
        try:
            self.fa_results = anal.run_fa_rank_analysis(X)
            self.fa_results['labels'] = labels
            self.current_plot_mode = 'FA_RANK_RESULTS'
            self.last_analysis_mode = 'FA_RANK_RESULTS'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd FA", f"{e}")

    # 4. Rekonstrukcja FA
    def _run_fa_reconstruction(self):
        if self.fa_results is None: return
        try:
            k = int(self.fa_recon_components_var.get()) # Upewnij się, że masz tę zmienną w __init__
            recon_res = anal.run_fa_reconstruction(self.fa_results, k)
            self.fa_results.update(recon_res)
            self.fa_results['recon_k'] = k
            self.current_plot_mode = 'FA_RECON_RESULTS'
            self.last_analysis_mode = 'FA_RECON_RESULTS'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd", f"{e}")

    # 5. SPEXFA
    def _run_spexfa(self):
        X, labels = self._get_selection_matrix()
        if X is None: return
        try:
            n = int(self.spexfa_n_components_var.get())
            self.spexfa_results = anal.run_spexfa(X, n)
            self.spexfa_results['labels'] = labels
            self.current_plot_mode = 'SPEXFA_RESULTS'
            self.last_analysis_mode = 'SPEXFA_RESULTS'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd SPEXFA", f"{e}")

    # 6. Rekonstrukcja Tensora (PARAFAC/MCR)
    def _run_tensor_recon(self, method):
        # method: 'parafac' lub 'mcr'
        data = self._get_active_data()
        if data is None: return
        try:
            k = int(self.tensor_recon_components_var.get())
            if method == 'parafac' and self.parafac_results:
                res = anal.run_parafac_reconstruction(self.parafac_results, data, k)
                self.tensor_recon_results = {**res, 'source': 'parafac', 'k': k}
            elif method == 'mcr' and self.mcr_results:
                res = anal.run_mcr_reconstruction(self.mcr_results, data, k)
                self.tensor_recon_results = {**res, 'source': 'mcr', 'k': k}
            else:
                return

            self.current_plot_mode = 'TENSOR_RECONSTRUCTION'
            self.last_analysis_mode = 'TENSOR_RECONSTRUCTION'
            self.update_plot()
        except Exception as e: messagebox.showerror("Błąd", f"{e}")


    # --- WIZUALIZACJA ---
    def _create_plot_canvas(self):
        self.fig = matplotlib.figure.Figure(figsize=(10, 7), dpi=100)
        self.axes = self.fig.subplots(2, 2)
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        
        # --- USUŃ TĘ LINIĘ (TO ONA BLOKOWAŁA RYSOWANIE): ---
        # self.canvas.get_tk_widget().bind("<Button-1>", lambda event: self.focus_set())
        # ---------------------------------------------------

        self.toolbar = matplotlib.backends.backend_tkagg.NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        # Zdarzenia Matplotlib
        self.canvas.mpl_connect('button_press_event', self._on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self._on_canvas_release)
    
    def update_plot(self):
        # 1. Czyścimy starą pamięć zoomu
        self.zoom_rects.clear()
        self.zoom_start.clear()
        self.initial_lims.clear() # Czyścimy zapamiętane limity

        self.fig.clear()
        mode = self.current_plot_mode
        
        # --- RYSOWANIE (Bez zmian logicznych, skopiuj swoją treść if/elif/else) ---
        # (Wklejam tu skrót, żeby nie zajmować miejsca - zachowaj swoją logikę rysowania!)
        
        if mode in ['SPECTRA', 'HEATMAP', 'PLS_RESULTS', 'MANIFOLD']:
            ax = self.fig.add_subplot(111)
            # ... (Twoja logika vis.plot_...)
            if mode == 'SPECTRA':
                data = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
                suffix = "(Preprocess)" if self.show_preprocessed_var.get() else "(Raw)"
                vis.plot_spectra(ax, self.wavenumbers, data, self.selected_coords, suffix)
            elif mode == 'HEATMAP':
                data = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
                vis.plot_heatmap(ax, data, self.wavenumbers, self.n_rows, self.m_cols)
            elif mode == 'PLS_RESULTS' and self.pls_results: vis.plot_pls(ax, self.pls_results, self.wavenumbers)
            elif mode == 'MANIFOLD' and self.manifold_results: vis.plot_manifold(ax, self.manifold_results)

        else:
            self.axes = self.fig.subplots(2, 2)
            # ... (Twoja logika vis.plot_... dla PCA, MCR, PARAFAC itp.)
            if mode == 'PCA' and self.pca_results: vis.plot_pca(self.axes, self.pca_results, self.wavenumbers)
            elif mode == 'MCR' and self.mcr_results: vis.plot_mcr(self.axes, self.mcr_results, self.wavenumbers, self.n_rows, self.m_cols)
            elif mode == 'PARAFAC' and self.parafac_results:
                 # (Skopiuj logikę PARAFAC z poprzedniej wersji)
                 factors = self.parafac_results['factors']
                 ax1, ax2, ax3, ax4 = self.axes.flat
                 for ax_temp in self.axes.flat: ax_temp.set_visible(True); vis.apply_theme(ax_temp)
                 k = factors[0].shape[1]
                 if self.wavenumbers is not None: ax1.plot(self.wavenumbers, factors[0]); vis.invert_xaxis_if_wavenumbers(ax1, self.wavenumbers)
                 ax1.set_title("PARAFAC: Widma")
                 ax2.plot(factors[1], 'o-'); ax2.set_title("Mode 1")
                 ax3.plot(factors[2], 'o-'); ax3.set_title("Mode 2")
                 ax4.bar(range(k), self.parafac_results['weights']); ax4.set_title("Wagi")
            elif mode == 'TUCKER_RESULTS' and self.tucker_results: vis.plot_tucker(self.axes, self.tucker_results, self.wavenumbers, self.n_rows, self.m_cols)
            elif mode == 'FA_RANK_RESULTS' and self.fa_results: vis.plot_fa_rank(self.axes, self.fa_results)
            elif mode == 'SPEXFA_RESULTS' and self.spexfa_results: vis.plot_spexfa(self.axes, self.spexfa_results, self.wavenumbers)
            elif mode in ['RECONSTRUCTION', 'FA_RECON_RESULTS', 'TENSOR_RECONSTRUCTION']:
                res = None
                if mode == 'RECONSTRUCTION': res = self.pca_results
                elif mode == 'FA_RECON_RESULTS': res = self.fa_results
                elif mode == 'TENSOR_RECONSTRUCTION': res = self.tensor_recon_results
                if res:
                    if 'source_tensor' not in res and 'X_original' not in res: res['source_tensor'] = self._get_active_data()
                    vis.plot_reconstruction(self.axes, res, self.wavenumbers, self.selected_coords)
            elif mode == '3DCOS_SLICER' and self.cos_3d_results:
                if hasattr(self, 'cos_slider_frame'): self.cos_slider_frame.pack(fill="x", pady=2)
                slice_idx = int(self.cos_slice_var.get())
                vis.plot_2dcos(self.axes, self.cos_3d_results, self.wavenumbers, slice_idx)

        if mode != '3DCOS_SLICER' and hasattr(self, 'cos_slider_frame'):
            self.cos_slider_frame.pack_forget()

        self.canvas.draw()
        
        # --- NOWOŚĆ: ZAPISZ STANY POCZĄTKOWE DLA RESETU ---
        # Iterujemy po wszystkich aktywnych osiach i zapisujemy ich limity
        for ax in self.fig.axes:
            if ax.get_visible():
                self.initial_lims[ax] = (ax.get_xlim(), ax.get_ylim())
        # --------------------------------------------------
        
        self.focus_set()

# --- RĘCZNA OBSŁUGA ZOOMU (NAPRAWIONA) ---
    
# --- POPRAWIONA OBSŁUGA ZOOMU (MAC OS COMPATIBLE) ---

    def _on_canvas_press(self, event):
        # 1. Najpierw ustaw focus (naprawa "zacinania", przeniesiona tutaj)
        self.focus_set()

        ax = event.inaxes
        if ax is None: return

        # Lewy przycisk: Start Zoom
        if event.button == 1:
            self.zoom_start[ax] = (event.xdata, event.ydata)
            
            # Tworzymy prostokąt z wypełnieniem (żeby był lepiej widoczny)
            rect = Rectangle(
                (event.xdata, event.ydata), 0, 0, 
                fill=True, facecolor='red', alpha=0.2, # Półprzezroczysty czerwony
                edgecolor='red', linestyle='-', linewidth=1
            )
            self.zoom_rects[ax] = rect
            ax.add_patch(rect)
            self.canvas.draw()

        # Prawy przycisk: Reset Zoom
        elif event.button == 3:
            if ax in self.initial_lims:
                xlim, ylim = self.initial_lims[ax]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                self.canvas.draw()
            else:
                ax.autoscale()
                self.canvas.draw()

    def _on_canvas_motion(self, event):
        # Rysujemy tylko, jeśli trzymamy LPM
        if event.button != 1: return

        # Szukamy aktywnego prostokąta dla tego wykresu
        # Używamy self.zoom_start jako źródła prawdy, a nie event.inaxes (które może zgubić się przy szybkim ruchu)
        active_ax = None
        for ax in self.zoom_start:
            # Sprawdzamy, czy ruch odbywa się w obrębie figury (nawet jeśli wyjedziemy poza oś)
            if ax.figure == self.fig:
                active_ax = ax
                break
        
        if active_ax and active_ax in self.zoom_rects:
            rect = self.zoom_rects[active_ax]
            x0, y0 = self.zoom_start[active_ax]
            
            # Pobieramy pozycję myszy (jeśli wyjedziemy poza, bierzemy krawędź)
            x1 = event.xdata
            y1 = event.ydata
            
            # Jeśli wyjechaliśmy myszą poza wykres, xdata/ydata mogą być None. 
            # Wtedy ignorujemy aktualizację lub bierzemy granice (tutaj prościej: ignorujemy)
            if x1 is not None and y1 is not None:
                width = x1 - x0
                height = y1 - y0
                rect.set_width(width)
                rect.set_height(height)
                rect.set_xy((x0, y0))
                
                # Używamy draw_idle zamiast draw dla płynności
                self.canvas.draw_idle() 

    def _on_canvas_release(self, event):
        if event.button == 1:
            # Znajdź aktywną oś na podstawie zapamiętanego startu
            active_ax = None
            for ax in self.zoom_start:
                active_ax = ax
                break
            
            if active_ax and active_ax in self.zoom_rects:
                # 1. Usuń prostokąt
                rect = self.zoom_rects[active_ax]
                rect.remove()
                del self.zoom_rects[active_ax]
                
                # 2. Pobierz współrzędne
                x0, y0 = self.zoom_start[active_ax]
                del self.zoom_start[active_ax]
                
                x1, y1 = event.xdata, event.ydata
                
                # Jeśli puściliśmy mysz poza oknem, spróbujmy użyć ostatniej znanej pozycji
                # (Dla uproszczenia: jeśli None, nie zoomujemy)
                if x1 is not None and y1 is not None:
                    # Minimalny próg ruchu (żeby nie zoomować przypadkowych kliknięć)
                    if abs(x1 - x0) > 1e-5 or abs(y1 - y0) > 1e-5:
                        
                        # Obsługa odwróconej osi (ważne dla IR!)
                        cur_xlim = active_ax.get_xlim()
                        if cur_xlim[0] > cur_xlim[1]:
                            active_ax.set_xlim(max(x0, x1), min(x0, x1))
                        else:
                            active_ax.set_xlim(min(x0, x1), max(x0, x1))
                        
                        active_ax.set_ylim(min(y0, y1), max(y0, y1))
                
                self.canvas.draw()
            
            # Sprzątanie awaryjne (jeśli coś zostało)
            self.zoom_rects.clear()
            self.zoom_start.clear()


    # --- EXPORT / SAVE ---
# --- main.py (Sekcja EXPORT / SAVE) ---

    def _save_project(self):
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if f:
            try:
                status_map = {f"{r},{c}": s for (r, c), s in self.field_matrix_status.items()}
                
                # Przekazujemy self.pipeline_steps do zapisu
                data_io.save_project_json(
                    f, 
                    self.n_rows, 
                    self.m_cols, 
                    self.original_wavenumbers, 
                    self.original_tensor_data, 
                    status_map,
                    self.pipeline_steps
                )
                messagebox.showinfo("Sukces", "Projekt został zapisany.")
            except Exception as e:
                messagebox.showerror("Błąd Zapisu", f"Nie udało się zapisać projektu:\n{e}")

    def _load_project(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if f:
            try:
                # 1. Wczytaj dane
                d = data_io.load_project_json(f)
                
                # 2. Ustaw wymiary i zresetuj siatkę
                self.n_var.set(d['n_rows'])
                self.m_var.set(d['m_cols'])
                self._create_field_matrix() # To czyści self.pipeline_steps, więc robimy to najpierw
                
                # 3. Przywróć dane tensora
                self.original_wavenumbers = d['original_wavenumbers']
                self.original_tensor_data = d['original_tensor_data']
                self._reset_wavenumber_range() # To ustawia self.tensor_data na podstawie oryginału
                
                # 4. Przywróć statusy komórek
                loaded_status = d['field_matrix_status']
                for str_coords, status in loaded_status.items():
                    r, c = map(int, str_coords.split(','))
                    coords = (r, c)
                    self._update_cell_status(coords, status, f"{coords}\n{status}")
                    if status in ['LOADED', 'FILLED']:
                        # Opcjonalnie: od razu zaznacz aktywne, żeby użytkownik je widział
                        pass

                # 5. PRZYWRÓĆ PIPELINE (Najważniejsza zmiana)
                saved_pipeline = d.get('pipeline_steps', [])
                if saved_pipeline:
                    self.pipeline_steps = saved_pipeline
                    self._refresh_pipeline_ui() # Odtwórz kafelki w GUI
                    self._run_pipeline()        # Przelicz dane
                
                self.update_plot()
                messagebox.showinfo("Sukces", "Projekt został wczytany.")
                
            except Exception as e:
                messagebox.showerror("Błąd Wczytywania", f"Nie udało się wczytać projektu:\n{e}")
                # Reset awaryjny
                self._create_field_matrix()

    def _export_to_xlsx(self):
        f = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if f:
            try:
                # Używamy getattr(self, 'nazwa', None) - to bezpieczne podejście.
                # Jeśli zmienna nie istnieje, wstawi None zamiast wyrzucać błąd.
                results = {
                    'pca': getattr(self, 'pca_results', None),
                    'mcr': getattr(self, 'mcr_results', None),
                    'parafac': getattr(self, 'parafac_results', None),
                    'tucker': getattr(self, 'tucker_results', None),
                    'spexfa': getattr(self, 'spexfa_results', None),
                    'fa': getattr(self, 'fa_results', None),
                    'pls': getattr(self, 'pls_results', None),
                    'manifold': getattr(self, 'manifold_results', None),
                    
                    # Rekonstrukcja - sprawdzamy bezpiecznie
                    'recon': getattr(self, 'tensor_recon_results', None) if getattr(self, 'tensor_recon_results', None) else 
                             (getattr(self, 'pca_results', None) if self.current_plot_mode == 'RECONSTRUCTION' else None)
                }

                data_io.export_results_to_xlsx(
                    f, 
                    self.wavenumbers, 
                    self.tensor_data, 
                    self.preprocessed_tensor, 
                    self.n_rows, self.m_cols, 
                    results,
                    getattr(self, 'pipeline_steps', None) # Też bezpiecznie
                )
                messagebox.showinfo("Sukces", "Wyeksportowano pomyślnie.")
            except Exception as e:
                # Wypisujemy pełny błąd do konsoli dla łatwiejszego debugowania
                print(f"BŁĄD SZCZEGÓŁOWY: {e}") 
                messagebox.showerror("Błąd Eksportu", f"Nie udało się wyeksportować danych:\n{e}")

    def _reset_analysis_results(self):
        self.pca_results = None
        self.mcr_results = None
        self.parafac_results = None
        self.manifold_results = None

    def _on_global_click(self, event):
        """
        Zabiera focus z przycisków po kliknięciu gdziekolwiek w tło aplikacji.
        Pozwala to uniknąć przypadkowego ponownego uruchomienia funkcji spacją/enterem.
        Nie zabiera focusu, jeśli kliknięto w pole tekstowe (Entry).
        """
        # Sprawdzamy, czy kliknięto w widget typu Entry (pole tekstowe)
        # CustomTkinter Entry zawiera w środku standardowe tkinter.Entry
        if isinstance(event.widget, tkinter.Entry):
            return # Nie rób nic, pozwól użytkownikowi pisać
        
        # W każdym innym przypadku (tło, etykiety, ramki) - zresetuj focus na główne okno
        self.focus_set()

if __name__ == "__main__":
    app = ChemTensorApp()
    app.mainloop()