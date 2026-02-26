import tkinter
from tkinter import filedialog, messagebox, Menu
import customtkinter as ctk
import matplotlib.figure
import matplotlib.backends.backend_tkagg
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import os

# --- IMPORTY NASZYCH MODUŁÓW ---
from modules import preprocessing as prep
from modules import analysis as anal
from modules import data_io
from modules import visualization as vis

# --- Status Color Constants ---
STATUS_COLORS = {
    'EMPTY': ctk.ThemeManager.theme["CTkButton"]["fg_color"],
    'MISSING': "gray50",
    'LOADED': "#2ECC71",
    'ERROR': "#E74C3C",
    'FILLED': "#0096FF"
}
SELECTED_BORDER_COLOR = "#3498DB"


class FileManagerWindow(ctk.CTkToplevel):
    def __init__(self, parent, file_list, callback_apply):
        super().__init__(parent)
        self.title("File Manager")
        self.geometry("600x500")

        self.parent = parent
        self.file_list = file_list
        self.callback_apply = callback_apply
        self.selected_index = None

        self.file_manager_window = None
        self.loaded_files_buffer = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.scroll_frame = ctk.CTkScrollableFrame(self, label_text="Loaded Files Sequence")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

        self._setup_controls()

        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        ctk.CTkButton(self.bottom_frame, text="Apply to Grid", command=self._on_apply, fg_color="#2ECC71",
                      hover_color="#27AE60").pack(fill="x", pady=5)

        self._refresh_list()

    def _setup_controls(self):
        btn_opts = {'width': 120}
        ctk.CTkButton(self.controls_frame, text="Add Files...", command=self._add_files, **btn_opts).pack(pady=5)
        ctk.CTkButton(self.controls_frame, text="Add Placeholder", command=self._add_placeholder, fg_color="#E67E22",
                      hover_color="#D35400", **btn_opts).pack(pady=5)
        ctk.CTkLabel(self.controls_frame, text="---").pack(pady=2)
        ctk.CTkButton(self.controls_frame, text="Move Up ▲", command=lambda: self._move_item(-1), **btn_opts).pack(
            pady=5)
        ctk.CTkButton(self.controls_frame, text="Move Down ▼", command=lambda: self._move_item(1), **btn_opts).pack(
            pady=5)
        ctk.CTkLabel(self.controls_frame, text="---").pack(pady=2)
        ctk.CTkButton(self.controls_frame, text="Sort A-Z", command=self._sort_az, **btn_opts).pack(pady=5)
        ctk.CTkButton(self.controls_frame, text="Remove Selected", command=self._remove_selected, fg_color="#C0392B",
                      hover_color="#E74C3C", **btn_opts).pack(pady=5)

    def _refresh_list(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        for i, item in enumerate(self.file_list):
            bg_color = "transparent"
            if i == self.selected_index:
                bg_color = ("#3B8ED0", "#1F6AA5")

            row = ctk.CTkFrame(self.scroll_frame, fg_color=bg_color)
            row.pack(fill="x", pady=1)
            row.bind("<Button-1>", lambda e, idx=i: self._select_item(idx))

            name = item['name']
            if item.get('is_placeholder'):
                name = "[ EMPTY PLACEHOLDER ]"
                text_color = "orange"
            else:
                text_color = "text_color"

            lbl = ctk.CTkLabel(row, text=f"{i + 1}. {name}", anchor="w",
                               text_color=text_color if text_color != "text_color" else None)
            lbl.pack(fill="x", padx=5, pady=2)
            lbl.bind("<Button-1>", lambda e, idx=i: self._select_item(idx))

    def _select_item(self, index):
        self.selected_index = index
        self._refresh_list()

    def _move_item(self, direction):
        if self.selected_index is None: return
        new_index = self.selected_index + direction

        if 0 <= new_index < len(self.file_list):
            self.file_list[self.selected_index], self.file_list[new_index] = \
                self.file_list[new_index], self.file_list[self.selected_index]
            self.selected_index = new_index
            self._refresh_list()

    def _remove_selected(self):
        if self.selected_index is None: return
        del self.file_list[self.selected_index]
        self.selected_index = None
        self._refresh_list()

    def _add_placeholder(self):
        self.file_list.append({'name': 'PLACEHOLDER', 'is_placeholder': True, 'data': None})
        self._refresh_list()

    def _add_files(self):
        paths = filedialog.askopenfilenames(title="Add Files")
        if not paths: return

        ref_wav = self.parent.original_wavenumbers
        for p in paths:
            try:
                wav, absorb = data_io.load_single_csv(p)
                if ref_wav is not None and not np.array_equal(ref_wav, wav):
                    messagebox.showwarning("Dimension Mismatch",
                                           f"File {os.path.basename(p)} has different wavenumbers. Skipped.")
                    continue

                if ref_wav is None:
                    self.parent.original_wavenumbers = wav
                    ref_wav = wav

                filename = os.path.basename(p)
                name_only = os.path.splitext(filename)[0]
                self.file_list.append({
                    'name': name_only,
                    'path': p,
                    'data': absorb,
                    'is_placeholder': False
                })
            except Exception as e:
                print(f"Error loading {p}: {e}")

        self._refresh_list()

    def _sort_az(self):
        self.file_list.sort(key=lambda x: x['name'])
        self._refresh_list()

    def _on_apply(self):
        self.callback_apply(self.file_list)
        self.destroy()


class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.id = None
        self.x = self.y = 0
        self._id_binding_enter = self.widget.bind('<Enter>', self.enter)
        self._id_binding_leave = self.widget.bind('<Leave>', self.leave)
        self._id_binding_press = self.widget.bind('<ButtonPress>', self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hide()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(1000, self.show)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def show(self):
        if self.tooltip_window:
            return

        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tw = tkinter.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))

        label = tkinter.Label(tw, text=self.text, justify='left',
                              background="#ffffe0", relief='solid', borderwidth=1,
                              font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide(self):
        tw = self.tooltip_window
        self.tooltip_window = None
        if tw:
            tw.destroy()


class ChemTensorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ChemTensor Explorer (Modular)")
        self.geometry("1400x1000")

        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

        # GUI Variables
        self.n_var = ctk.StringVar(value="4")
        self.m_var = ctk.StringVar(value="4")
        self.range_min_var = ctk.StringVar()
        self.range_max_var = ctk.StringVar()
        self.file_manager_window = None
        self.loaded_files_buffer = []

        # Preprocessing vars
        self.sg_window_var = ctk.StringVar(value="11")
        self.sg_poly_var = ctk.StringVar(value="3")
        self.sg_smooth_window_var = ctk.StringVar(value="11")
        self.sg_smooth_poly_var = ctk.StringVar(value="3")
        self.als_lambda_var = ctk.StringVar(value="1e6")
        self.als_p_var = ctk.StringVar(value="0.01")
        self.offset_target_val_var = ctk.StringVar(value="0.0")
        self.offset_point_var = ctk.StringVar(value="")
        self.show_preprocessed_var = ctk.BooleanVar(value=False)
        self.pipeline_steps = []

        # Analysis vars
        self.pca_n_components_var = ctk.StringVar(value="2")
        self.pca_recon_components_var = ctk.StringVar(value="2")
        self.fa_recon_components_var = ctk.StringVar(value="2")
        self.parafac_rank_var = ctk.StringVar(value="2")
        self.parafac_non_negative_var = ctk.BooleanVar(value=False)

        # MCR specific vars
        self.mcr_n_components_var = ctk.StringVar(value="2")
        self.mcr_non_negative_var = ctk.BooleanVar(value=True)
        self.mcr_norm_var = ctk.BooleanVar(value=False)
        self.mcr_spexfa_init_var = ctk.BooleanVar(value=True)
        self.mcr_closure_var = ctk.BooleanVar(value=False)
        self.mcr_unimodal_var = ctk.BooleanVar(value=False)
        self.mcr_max_iter_var = ctk.StringVar(value="500")
        self.mcr_tol_var = ctk.StringVar(value="1e-4")
        self.mcr_st_fix_var = ctk.StringVar(value="")

        self.tucker_rank_w_var = ctk.StringVar(value="2")
        self.tucker_rank_n_var = ctk.StringVar(value="2")
        self.tucker_rank_m_var = ctk.StringVar(value="2")
        self.tensor_recon_components_var = ctk.StringVar(value="2")
        self.spexfa_n_components_var = ctk.StringVar(value="2")
        self.pls_target_var = ctk.StringVar(value="Select Target (y)...")
        self.cos_axis_var = ctk.StringVar(value="Analyze Rows (N)")
        self.cos_slice_var = ctk.DoubleVar(value=0)

        # Data State
        self.field_matrix_widgets = {}
        self.field_matrix_status = {}
        self.selected_coords = set()

        self.original_tensor_data = None
        self.original_wavenumbers = None
        self.tensor_data = None
        self.wavenumbers = None
        self.preprocessed_tensor = None

        # Analysis Results
        self.pca_results = None
        self.parafac_results = None
        self.mcr_results = None
        self.tucker_results = None
        self.spexfa_results = None
        self.pls_results = None
        self.fa_results = None
        self.tensor_recon_results = None
        self.cos_3d_results = None
        self.manifold_results = None
        self.mcr_st_init = None

        self.current_plot_mode = 'SPECTRA'
        self.last_analysis_mode = None

        # Plot Tools
        self.zoom_rects = {}
        self.zoom_start = {}
        self.initial_lims = {}
        self.pls_target_map = {}

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

        self.tab_view = ctk.CTkTabview(self.control_frame, command=self._on_tab_change)
        self.tab_view.grid(row=0, column=0, sticky="nsew")

        data_tab = self.tab_view.add("Data")
        preprocess_tab = self.tab_view.add("Preprocessing")
        analysis_tab = self.tab_view.add("Analysis")

        self._setup_data_tab(data_tab)
        self._setup_preprocess_tab(preprocess_tab)
        self._setup_analysis_tab(analysis_tab)

    def _on_tab_change(self):
        tab_name = self.tab_view.get()

        if tab_name == "Data":
            self.show_preprocessed_var.set(False)
            self.current_plot_mode = 'SPECTRA'
        elif tab_name == "Preprocessing":
            self.show_preprocessed_var.set(True)
            self.current_plot_mode = 'SPECTRA'
        elif tab_name == "Analysis":
            if self.last_analysis_mode:
                self.current_plot_mode = self.last_analysis_mode
            else:
                self.current_plot_mode = 'SPECTRA'
        self.update_plot()

    def _setup_data_tab(self, parent):
        frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        frame.pack(fill="both", expand=True)

        row0 = ctk.CTkFrame(frame)
        row0.pack(fill="x", pady=5)
        ctk.CTkLabel(row0, text="R:").pack(side="left", padx=5)
        ctk.CTkEntry(row0, textvariable=self.n_var, width=50).pack(side="left")
        ctk.CTkLabel(row0, text="C:").pack(side="left", padx=5)
        ctk.CTkEntry(row0, textvariable=self.m_var, width=50).pack(side="left")
        ctk.CTkButton(row0, text="Create Grid", command=self.safe_create_field_matrix).pack(side="left", padx=10)

        ctk.CTkLabel(frame, text="Field Matrix").pack(anchor="w")
        self.field_matrix_frame = ctk.CTkFrame(frame)
        self.field_matrix_frame.pack(fill="x", pady=5)

        row2 = ctk.CTkFrame(frame)
        row2.pack(fill="x", pady=5)
        ctk.CTkButton(row2, text="Manage Files", command=self._open_file_manager).pack(side="left", fill="x",
                                                                                       expand=True, padx=2)
        ctk.CTkButton(row2, text="Select Active", command=self._select_all_active).pack(side="left", fill="x",
                                                                                        expand=True, padx=2)
        ctk.CTkButton(row2, text="Deselect", command=self._deselect_all).pack(side="left", fill="x", expand=True,
                                                                              padx=2)

        row3 = ctk.CTkFrame(frame)
        row3.pack(fill="x", pady=5)

        ctk.CTkLabel(row3, text="Range (cm-1):").pack(side="left", padx=(5, 2))
        ctk.CTkEntry(row3, textvariable=self.range_min_var, width=60).pack(side="left", padx=2)
        ctk.CTkLabel(row3, text="-").pack(side="left", padx=2)
        ctk.CTkEntry(row3, textvariable=self.range_max_var, width=60).pack(side="left", padx=2)

        buttons_frame = ctk.CTkFrame(row3, fg_color="transparent")
        buttons_frame.pack(side="left", fill="x", expand=True, padx=5)

        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(buttons_frame, text="Apply", command=self._apply_wavenumber_range).grid(row=0, column=0,
                                                                                              sticky="ew", padx=2)
        ctk.CTkButton(buttons_frame, text="Reset", command=self._reset_wavenumber_range).grid(row=0, column=1,
                                                                                              sticky="ew", padx=2)

        row4 = ctk.CTkFrame(frame)
        row4.pack(fill="x", pady=10)
        ctk.CTkButton(row4, text="Save Project", command=self._save_project).pack(fill="x", pady=2)
        ctk.CTkButton(row4, text="Load Project", command=self._load_project).pack(fill="x", pady=2)
        ctk.CTkButton(row4, text="Export to Excel", command=self._export_to_xlsx).pack(fill="x", pady=2)

    def _open_file_manager(self):
        if self.file_manager_window is None or not self.file_manager_window.winfo_exists():
            self.file_manager_window = FileManagerWindow(self, self.loaded_files_buffer, self._apply_file_list_to_grid)
        else:
            self.file_manager_window.focus()

    def _apply_file_list_to_grid(self, file_list):
        self.loaded_files_buffer = file_list
        n, m = self.n_rows, self.m_cols
        total_cells = n * m
        if not file_list: return

        if self.original_tensor_data is None:
            first_valid = next((f for f in file_list if not f.get('is_placeholder')), None)
            if first_valid:
                wav_len = len(first_valid['data'])
                self.original_tensor_data = np.full((wav_len, n, m), np.nan)
            else:
                return

        idx = 0
        for r in range(n):
            for c in range(m):
                coords = (r, c)
                if idx >= len(file_list):
                    self._mark_cell_as_empty(coords)
                else:
                    item = file_list[idx]
                    if item.get('is_placeholder'):
                        self._mark_cell_as_missing(coords)
                        self._update_cell_status(coords, 'MISSING', f"{coords}\n[SKIP]")
                    else:
                        absorb = item['data']
                        name = item['name']
                        suffix_len = 6
                        display_name = ".." + name[-suffix_len:] if len(name) > suffix_len else name
                        self.original_tensor_data[:, r, c] = absorb
                        self._update_cell_status(coords, 'LOADED', f"{coords}\n{display_name}")
                        if coords in self.field_matrix_widgets:
                            ToolTip(self.field_matrix_widgets[coords], text=item.get('name', ''))
                idx += 1

        self._reset_wavenumber_range()
        self.update_plot()
        messagebox.showinfo("Success", f"Mapped {min(len(file_list), total_cells)} files to grid.")

    def _setup_visual_tab(self, parent):
        frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        ctk.CTkSwitch(frame, text="Show processed data", variable=self.show_preprocessed_var,
                      command=self.update_plot).pack(pady=10)
        ctk.CTkButton(frame, text="Show Heatmap", command=self._run_heatmap).pack(fill="x", pady=5)

    def _setup_preprocess_tab(self, parent):
        main_frame = ctk.CTkFrame(parent, fg_color="transparent")
        main_frame.pack(fill="both", expand=True)

        buttons_frame = ctk.CTkScrollableFrame(main_frame, height=400)
        buttons_frame.pack(fill="x", expand=False, padx=5, pady=5)

        ctk.CTkSwitch(buttons_frame, text="Show processed data (Preview)",
                      variable=self.show_preprocessed_var,
                      command=self.update_plot).pack(pady=(5, 5), anchor="w")

        # 1. ALS
        als_grp = ctk.CTkFrame(buttons_frame)
        als_grp.pack(fill="x", pady=5)
        ctk.CTkLabel(als_grp, text="ALS Baseline", font=("Arial", 12, "bold")).pack()

        row_als = ctk.CTkFrame(als_grp, fg_color="transparent")
        row_als.pack(fill="x")
        lbl_als_lam = ctk.CTkLabel(row_als, text="λ:", width=20)
        lbl_als_lam.pack(side="left", padx=(5, 0))
        ToolTip(lbl_als_lam, text="Smoothness (Lambda).\nHigher value (e.g., 1e5 - 1e7) = smoother/stiffer baseline.")
        ctk.CTkEntry(row_als, textvariable=self.als_lambda_var, width=50).pack(side="left", padx=2)
        lbl_als_p = ctk.CTkLabel(row_als, text="p:", width=20)
        lbl_als_p.pack(side="left", padx=(5, 0))
        ToolTip(lbl_als_p, text="Asymmetry (p).\nWeight for positive residuals (0 < p < 1).")
        ctk.CTkEntry(row_als, textvariable=self.als_p_var, width=40).pack(side="left", padx=2)
        ctk.CTkButton(row_als, text="Add", command=self._apply_als).pack(side="left", fill="x", expand=True, padx=5)

        # 1.5 Offset
        off_grp = ctk.CTkFrame(buttons_frame)
        off_grp.pack(fill="x", pady=5)
        ctk.CTkLabel(off_grp, text="Offset Correction", font=("Arial", 12, "bold")).pack()
        row_off = ctk.CTkFrame(off_grp, fg_color="transparent")
        row_off.pack(fill="x")
        ctk.CTkLabel(row_off, text="Set").pack(side="left", padx=(5, 2))
        ctk.CTkEntry(row_off, textvariable=self.offset_target_val_var, width=40).pack(side="left", padx=2)
        ctk.CTkLabel(row_off, text="at").pack(side="left", padx=2)
        entry_wn = ctk.CTkEntry(row_off, textvariable=self.offset_point_var, width=50, placeholder_text="cm-1")
        entry_wn.pack(side="left", padx=2)
        ToolTip(entry_wn, text="Wavenumber point to anchor the spectrum.")
        ctk.CTkLabel(row_off, text="cm⁻¹").pack(side="left", padx=(0, 5))
        ctk.CTkButton(row_off, text="Add", command=self._apply_offset).pack(side="left", fill="x", expand=True, padx=5)

        # 2. Normalization
        norm_grp = ctk.CTkFrame(buttons_frame)
        norm_grp.pack(fill="x", pady=5)
        ctk.CTkLabel(norm_grp, text="Normalization", font=("Arial", 12, "bold")).pack()
        grid_norm = ctk.CTkFrame(norm_grp, fg_color="transparent")
        grid_norm.pack(fill="x")
        ctk.CTkButton(grid_norm, text="SNV", command=self._apply_snv, width=60).grid(row=0, column=0, padx=2, pady=2,
                                                                                     sticky="ew")
        ctk.CTkButton(grid_norm, text="Min-Max", command=self._apply_min_max, width=60).grid(row=0, column=1, padx=2,
                                                                                             pady=2, sticky="ew")
        ctk.CTkButton(grid_norm, text="Area", command=self._apply_l1_norm, width=60).grid(row=0, column=2, padx=2,
                                                                                          pady=2, sticky="ew")
        ctk.CTkButton(grid_norm, text="MSC", command=self._apply_msc, width=60).grid(row=0, column=3, padx=2, pady=2,
                                                                                     sticky="ew")
        for i in range(4): grid_norm.grid_columnconfigure(i, weight=1)

        # 3. Smoothing
        smooth_grp = ctk.CTkFrame(buttons_frame)
        smooth_grp.pack(fill="x", pady=5)
        ctk.CTkLabel(smooth_grp, text="Smoothing (Savitzky-Golay)", font=("Arial", 12, "bold")).pack()
        row_smooth = ctk.CTkFrame(smooth_grp, fg_color="transparent")
        row_smooth.pack(fill="x", pady=2)
        lbl_sm_w = ctk.CTkLabel(row_smooth, text="w:", width=20)
        lbl_sm_w.pack(side="left", padx=(5, 0))
        ToolTip(lbl_sm_w, text="Window length")
        ctk.CTkEntry(row_smooth, textvariable=self.sg_smooth_window_var, width=40).pack(side="left", padx=2)
        lbl_sm_p = ctk.CTkLabel(row_smooth, text="p:", width=20)
        lbl_sm_p.pack(side="left", padx=(5, 0))
        ctk.CTkEntry(row_smooth, textvariable=self.sg_smooth_poly_var, width=40).pack(side="left", padx=2)
        ctk.CTkButton(row_smooth, text="Add", command=self._apply_sg_smoothing).pack(side="left", fill="x", expand=True,
                                                                                     padx=5)

        # 4. 2nd Deriv.
        sg_grp = ctk.CTkFrame(buttons_frame)
        sg_grp.pack(fill="x", pady=5)
        ctk.CTkLabel(sg_grp, text="2nd Deriv. (Savitzky-Golay)", font=("Arial", 12, "bold")).pack()
        row_sg = ctk.CTkFrame(sg_grp, fg_color="transparent")
        row_sg.pack(fill="x", pady=2)
        lbl_der_w = ctk.CTkLabel(row_sg, text="w:", width=20)
        lbl_der_w.pack(side="left", padx=(5, 0))
        ctk.CTkEntry(row_sg, textvariable=self.sg_window_var, width=40).pack(side="left", padx=2)
        lbl_der_p = ctk.CTkLabel(row_sg, text="p:", width=20)
        lbl_der_p.pack(side="left", padx=(5, 0))
        ctk.CTkEntry(row_sg, textvariable=self.sg_poly_var, width=40).pack(side="left", padx=2)
        ctk.CTkButton(row_sg, text="Add", command=self._apply_sg_filter).pack(side="left", fill="x", expand=True,
                                                                              padx=5)

        ctk.CTkLabel(main_frame, text="Active Steps (Order):", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.pipeline_scroll = ctk.CTkScrollableFrame(main_frame, height=200,
                                                      fg_color="gray90" if ctk.get_appearance_mode() == "Light" else "gray20")
        self.pipeline_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        ctk.CTkLabel(self.pipeline_scroll, text="(No steps - raw data)", text_color="gray50").pack(pady=20)

    def _setup_analysis_tab(self, parent):
        frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        frame.pack(fill="both", expand=True)

        # 1. FA / SPEXFA
        fa_frame = ctk.CTkFrame(frame)
        fa_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(fa_frame, text="Factor Analysis (Malinowski / SPEXFA)", font=("Arial", 12, "bold")).pack()
        ctk.CTkButton(fa_frame, text="Rank Analysis (RE/IND)", command=self._run_fa_rank_analysis).pack(fill="x",
                                                                                                        pady=2)
        row_spex = ctk.CTkFrame(fa_frame, fg_color="transparent")
        row_spex.pack(fill="x", pady=2)
        ctk.CTkEntry(row_spex, textvariable=self.spexfa_n_components_var, width=40).pack(side="left")
        ctk.CTkButton(row_spex, text="Run SPEXFA", command=self._run_spexfa).pack(side="left", fill="x", expand=True)

        row_fa_rec = ctk.CTkFrame(fa_frame, fg_color="transparent")
        row_fa_rec.pack(fill="x", pady=2)
        ctk.CTkEntry(row_fa_rec, textvariable=self.fa_recon_components_var, width=40).pack(side="left")
        ctk.CTkButton(row_fa_rec, text="FA Reconstruction", command=self._run_fa_reconstruction).pack(side="left",
                                                                                                      fill="x",
                                                                                                      expand=True)

        # 2. PCA
        pca_frame = ctk.CTkFrame(frame)
        pca_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(pca_frame, text="PCA (Principal Components)", font=("Arial", 12, "bold")).pack()
        row_pca = ctk.CTkFrame(pca_frame, fg_color="transparent")
        row_pca.pack(fill="x", pady=2)
        ctk.CTkEntry(row_pca, textvariable=self.pca_n_components_var, width=40).pack(side="left")
        ctk.CTkButton(row_pca, text="Calculate PCA", command=self._run_pca).pack(side="left", fill="x", expand=True)

        row_pca_rec = ctk.CTkFrame(pca_frame, fg_color="transparent")
        row_pca_rec.pack(fill="x", pady=2)
        ctk.CTkEntry(row_pca_rec, textvariable=self.pca_recon_components_var, width=40).pack(side="left")
        ctk.CTkButton(row_pca_rec, text="PCA Reconstruction", command=self._run_pca_reconstruction).pack(side="left",
                                                                                                         fill="x",
                                                                                                         expand=True)

        # 3. MCR-ALS (ULEPSZONE)
        mcr_frame = ctk.CTkFrame(frame)
        mcr_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(mcr_frame, text="MCR-ALS", font=("Arial", 12, "bold")).pack()

        row_mcr1 = ctk.CTkFrame(mcr_frame, fg_color="transparent")
        row_mcr1.pack(fill="x", pady=2)
        ctk.CTkEntry(row_mcr1, textvariable=self.mcr_n_components_var, width=40).pack(side="left")
        ctk.CTkCheckBox(row_mcr1, text="NNLS", variable=self.mcr_non_negative_var).pack(side="left", padx=5)
        ctk.CTkButton(row_mcr1, text="Start MCR", command=self._run_mcr_als).pack(side="left", fill="x", expand=True)

        # Dodatkowe opcje restrykcji
        row_mcr_opts = ctk.CTkFrame(mcr_frame, fg_color="transparent")
        row_mcr_opts.pack(fill="x", pady=2)
        ctk.CTkCheckBox(row_mcr_opts, text="SPEXFA Init", variable=self.mcr_spexfa_init_var).pack(side="left", padx=5)
        ctk.CTkCheckBox(row_mcr_opts, text="Closure", variable=self.mcr_closure_var).pack(side="left", padx=5)
        ctk.CTkCheckBox(row_mcr_opts, text="Unimodal", variable=self.mcr_unimodal_var).pack(side="left", padx=5)

        # Kontrola konwergencji
        row_mcr_params = ctk.CTkFrame(mcr_frame, fg_color="transparent")
        row_mcr_params.pack(fill="x", pady=2)
        ctk.CTkLabel(row_mcr_params, text="Max Iter:").pack(side="left", padx=(5, 2))
        ctk.CTkEntry(row_mcr_params, textvariable=self.mcr_max_iter_var, width=40).pack(side="left")
        ctk.CTkLabel(row_mcr_params, text="Tol:").pack(side="left", padx=(10, 2))
        ctk.CTkEntry(row_mcr_params, textvariable=self.mcr_tol_var, width=50).pack(side="left")

        row_mcr2 = ctk.CTkFrame(mcr_frame, fg_color="transparent")
        row_mcr2.pack(fill="x", pady=2)
        ctk.CTkButton(row_mcr2, text="Known Spectra...", command=self._load_mcr_st_init).pack(side="left", fill="x",
                                                                                              expand=True)
        ctk.CTkEntry(row_mcr2, textvariable=self.mcr_st_fix_var, placeholder_text="Fix (e.g. 0,2)").pack(side="left",
                                                                                                         fill="x",
                                                                                                         expand=True)

        # 4. TENSOR
        tens_frame = ctk.CTkFrame(frame)
        tens_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(tens_frame, text="Tensor (PARAFAC / Tucker)", font=("Arial", 12, "bold")).pack()
        row_par = ctk.CTkFrame(tens_frame, fg_color="transparent")
        row_par.pack(fill="x", pady=2)
        ctk.CTkEntry(row_par, textvariable=self.parafac_rank_var, width=40).pack(side="left")
        ctk.CTkButton(row_par, text="PARAFAC", command=self._run_parafac).pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(tens_frame, text="Tucker Ranks (W, N, M):", font=("Arial", 10)).pack(anchor="w")
        row_tuck = ctk.CTkFrame(tens_frame, fg_color="transparent")
        row_tuck.pack(fill="x", pady=2)
        ctk.CTkEntry(row_tuck, textvariable=self.tucker_rank_w_var, width=30).pack(side="left")
        ctk.CTkEntry(row_tuck, textvariable=self.tucker_rank_n_var, width=30).pack(side="left")
        ctk.CTkEntry(row_tuck, textvariable=self.tucker_rank_m_var, width=30).pack(side="left")
        ctk.CTkButton(row_tuck, text="Tucker", command=self._run_tucker).pack(side="left", fill="x", expand=True)

        row_tens_rec = ctk.CTkFrame(tens_frame, fg_color="transparent")
        row_tens_rec.pack(fill="x", pady=5)
        ctk.CTkLabel(row_tens_rec, text="Reconstruction: ").pack(side="left")
        ctk.CTkEntry(row_tens_rec, textvariable=self.tensor_recon_components_var, width=30).pack(side="left")
        ctk.CTkButton(row_tens_rec, text="from PARAFAC", command=lambda: self._run_tensor_recon('parafac')).pack(
            side="left", padx=2)
        ctk.CTkButton(row_tens_rec, text="from MCR", command=lambda: self._run_tensor_recon('mcr')).pack(side="left",
                                                                                                         padx=2)

        # 5. 2D-COS
        cos_frame = ctk.CTkFrame(frame)
        cos_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(cos_frame, text="2D-COS", font=("Arial", 12, "bold")).pack()
        ctk.CTkOptionMenu(cos_frame, variable=self.cos_axis_var,
                          values=["Analyze Rows (N)", "Analyze Columns (M)"]).pack(fill="x", pady=2)
        ctk.CTkButton(cos_frame, text="Run 3D-COS", command=self._run_3dcos).pack(fill="x")

        self.cos_slider_frame = ctk.CTkFrame(cos_frame, fg_color="transparent")
        self.cos_slider_frame.pack(fill="x")
        ctk.CTkLabel(self.cos_slider_frame, text="Slice:").pack(side="left")
        self.cos_slider = ctk.CTkSlider(self.cos_slider_frame, from_=0, to=1, number_of_steps=1,
                                        command=self._on_cos_slider)
        self.cos_slider.pack(side="left", fill="x", expand=True)
        self.cos_slider_frame.pack_forget()

        # 6. MANIFOLD
        man_frame = ctk.CTkFrame(frame)
        man_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(man_frame, text="Dimensionality Reduction", font=("Arial", 12, "bold")).pack()
        row_man = ctk.CTkFrame(man_frame, fg_color="transparent")
        row_man.pack(fill="x", pady=2)
        ctk.CTkButton(row_man, text="UMAP", command=self._run_umap).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row_man, text="t-SNE", command=self._run_tsne).pack(side="left", fill="x", expand=True)

    def safe_create_field_matrix(self):
        if self.original_tensor_data is not None:
            if not messagebox.askyesno("Confirmation", "Creating a new grid will delete current data. Continue?"):
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
                cell_text = f"({r_idx},{c_idx})\nEmpty"
                btn = ctk.CTkButton(self.field_matrix_frame, text=cell_text,
                                    height=60, fg_color=STATUS_COLORS['EMPTY'])
                btn.grid(row=r_idx, column=c_idx, padx=1, pady=1, sticky="nsew")

                btn.bind("<Button-3>", lambda e, c=coords: self._on_cell_right_click(e, c))
                btn.bind("<Button-2>", lambda e, c=coords: self._on_cell_right_click(e, c))
                btn.bind("<Control-Button-1>", lambda e, c=coords: self._on_cell_right_click(e, c))

                btn.configure(command=lambda c=coords: self._on_cell_left_click(c))
                self.field_matrix_widgets[coords] = btn
                self.field_matrix_status[coords] = 'EMPTY'

    def _mark_cell_as_missing(self, coords):
        self._update_cell_status(coords, 'MISSING', f"{coords}\nMISSING")
        if self.original_tensor_data is not None:
            self.original_tensor_data[:, coords[0], coords[1]] = np.nan
        self._reset_wavenumber_range()

    def _mark_cell_as_empty(self, coords):
        self._update_cell_status(coords, 'EMPTY', f"{coords}\nEmpty")
        if self.original_tensor_data is not None:
            self.original_tensor_data[:, coords[0], coords[1]] = np.nan
        self._reset_wavenumber_range()

    def _fill_cell_with_zeros(self, coords):
        if self.original_tensor_data is None: return
        r, c = coords
        zeros = np.zeros(len(self.original_wavenumbers))
        self.original_tensor_data[:, r, c] = zeros

        self._update_cell_status(coords, 'FILLED_ZERO', f"{coords}\nFilled (0)")
        self._reset_wavenumber_range()
        self.update_plot()

    def _fill_cell_with_row_mean(self, coords):
        if self.original_tensor_data is None: return
        r, c = coords
        try:
            mean_spectrum = np.nanmean(self.original_tensor_data[:, r, :], axis=1)
            if np.all(np.isnan(mean_spectrum)): raise ValueError("No data in this row.")
            self.original_tensor_data[:, r, c] = mean_spectrum
            self._update_cell_status(coords, 'FILLED_ROW_MEAN', f"{coords}\nFilled (Row Mean)")
            self._reset_wavenumber_range()
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Cannot calculate row mean:\n{e}")

    def _fill_cell_with_col_mean(self, coords):
        if self.original_tensor_data is None: return
        r, c = coords
        try:
            mean_spectrum = np.nanmean(self.original_tensor_data[:, :, c], axis=1)
            if np.all(np.isnan(mean_spectrum)): raise ValueError("No data in this column.")
            self.original_tensor_data[:, r, c] = mean_spectrum
            self._update_cell_status(coords, 'FILLED_COL_MEAN', f"{coords}\nFilled (Col Mean)")
            self._reset_wavenumber_range()
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Cannot calculate column mean:\n{e}")

    def _fill_cell_with_row_neighbors(self, coords):
        if self.original_tensor_data is None: return
        r, c = coords
        neighbors = []
        if c > 0: neighbors.append(self.original_tensor_data[:, r, c - 1])
        if c < self.m_cols - 1: neighbors.append(self.original_tensor_data[:, r, c + 1])
        self._calculate_neighbors_mean(coords, neighbors, "Neighbors (Row)")

    def _fill_cell_with_col_neighbors(self, coords):
        if self.original_tensor_data is None: return
        r, c = coords
        neighbors = []
        if r > 0: neighbors.append(self.original_tensor_data[:, r - 1, c])
        if r < self.n_rows - 1: neighbors.append(self.original_tensor_data[:, r + 1, c])
        self._calculate_neighbors_mean(coords, neighbors, "Neighbors (Col)")

    def _calculate_neighbors_mean(self, coords, neighbors, label):
        try:
            valid_neighbors = [n for n in neighbors if not np.all(np.isnan(n))]
            if not valid_neighbors: raise ValueError("No valid neighbors (all empty or NaN).")
            mean_spec = np.nanmean(np.array(valid_neighbors), axis=0)
            self.original_tensor_data[:, coords[0], coords[1]] = mean_spec
            self._update_cell_status(coords, 'FILLED_NEIGHBOR', f"{coords}\n{label}")
            self._reset_wavenumber_range()
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Interpolation Error", f"Cannot fill from neighbors:\n{e}")

    def _load_data_files(self):
        cells = [c for c, s in self.field_matrix_status.items() if s == 'EMPTY']
        if not cells: return
        paths = filedialog.askopenfilenames(title=f"Select {len(cells)} CSV files")
        if len(paths) != len(cells):
            messagebox.showerror("Error", f"Selected {len(paths)} files, expected {len(cells)}")
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
                    filename = os.path.basename(path)
                    name_only = os.path.splitext(filename)[0]
                    suffix_len = 6
                    display_name = ".." + name_only[-suffix_len:] if len(name_only) > suffix_len else name_only
                    self._update_cell_status(coords, 'LOADED', f"{coords}\n{display_name}")
                    btn_widget = self.field_matrix_widgets[coords]
                    ToolTip(btn_widget, text=filename)
                else:
                    self._update_cell_status(coords, 'ERROR', "ERROR X")
            except Exception:
                self._update_cell_status(coords, 'ERROR', "FILE ERROR")
        if self.original_tensor_data is None: self.original_tensor_data = temp_data
        self._reset_wavenumber_range()
        self.update_plot()

    def _update_cell_status(self, coords, status, text):
        self.field_matrix_status[coords] = status
        btn = self.field_matrix_widgets[coords]
        btn.configure(text=text, fg_color=STATUS_COLORS.get(status.split('_')[0], 'gray'))

    def _on_cell_left_click(self, coords):
        status = self.field_matrix_status.get(coords)
        if status != 'LOADED' and not (status and status.startswith('FILLED')): return
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
        if status != 'LOADED':
            menu.add_command(label="Mark as MISSING", command=lambda: self._mark_cell_as_missing(coords))
            menu.add_command(label="Mark as EMPTY", command=lambda: self._mark_cell_as_empty(coords))
        if self.original_tensor_data is not None:
            menu.add_separator()
            fill_menu = Menu(menu, tearoff=0)
            menu.add_cascade(label="Fill with data...", menu=fill_menu)
            fill_menu.add_command(label="Zeros (0.0)", command=lambda: self._fill_cell_with_zeros(coords))
            fill_menu.add_separator()
            fill_menu.add_command(label="ROW Mean (Entire)", command=lambda: self._fill_cell_with_row_mean(coords))
            fill_menu.add_command(label="COLUMN Mean (Entire)", command=lambda: self._fill_cell_with_col_mean(coords))
            fill_menu.add_separator()
            fill_menu.add_command(label="Neighbor Mean (Left/Right)",
                                  command=lambda: self._fill_cell_with_row_neighbors(coords))
            fill_menu.add_command(label="Neighbor Mean (Top/Bottom)",
                                  command=lambda: self._fill_cell_with_col_neighbors(coords))
        if status.startswith('LOADED') or status.startswith('FILLED'):
            menu.add_separator()
            menu.add_command(label="Reset (Mark as Empty)", command=lambda: self._mark_cell_as_empty(coords))
        menu.post(event.x_root, event.y_root)

    def _select_all_active(self):
        for coords, status in self.field_matrix_status.items():
            is_filled = status and status.startswith('FILLED')
            if (status == 'LOADED' or is_filled) and coords not in self.selected_coords:
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
            if self.pipeline_steps:
                self._run_pipeline()
            else:
                self.update_plot()
        except ValueError:
            messagebox.showerror("Error", "Invalid range")

    def _reset_wavenumber_range(self):
        if self.original_tensor_data is not None:
            self.wavenumbers = np.copy(self.original_wavenumbers)
            self.tensor_data = np.copy(self.original_tensor_data)
            self.range_min_var.set(f"{np.min(self.wavenumbers):.2f}")
            self.range_max_var.set(f"{np.max(self.wavenumbers):.2f}")

            self.preprocessed_tensor = None
            if self.pipeline_steps:
                self._run_pipeline()
            else:
                self.update_plot()

    def _get_active_data(self):
        if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None:
            return self.preprocessed_tensor
        return self.tensor_data

    def _process_data(self, func, name):
        data = self._get_active_data()
        if data is None: return
        try:
            new_tensor = func(data)
            self.preprocessed_tensor = new_tensor
            self.show_preprocessed_var.set(True)
            self.update_plot()
            messagebox.showinfo("Success", f"Applied: {name}")
        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def _apply_sg_smoothing(self):
        try:
            w = int(self.sg_smooth_window_var.get())
            p = int(self.sg_smooth_poly_var.get())
            label = f"Smoothing (w={w}, p={p})"
            self._add_pipeline_step('SG', {'w': w, 'poly': p, 'deriv': 0}, label)
        except ValueError:
            messagebox.showerror("Error", "Integers required")

    def _apply_sg_filter(self):
        try:
            w = int(self.sg_window_var.get())
            p = int(self.sg_poly_var.get())
            label = f"2nd Deriv (w={w}, p={p})"
            self._add_pipeline_step('SG', {'w': w, 'poly': p, 'deriv': 2}, label)
        except ValueError:
            messagebox.showerror("Error", "Integers required")

    def _apply_snv(self):
        self._add_pipeline_step('SNV', label="SNV")

    def _apply_min_max(self):
        self._add_pipeline_step('MinMax', label="Min-Max")

    def _apply_l1_norm(self):
        self._add_pipeline_step('L1', label="L1 Norm")

    def _apply_msc(self):
        self._add_pipeline_step('MSC', label="MSC")

    def _apply_als(self):
        try:
            lam = float(self.als_lambda_var.get())
            p = float(self.als_p_var.get())
            label = f"ALS (λ={lam:.0e}, p={p})"
            self._add_pipeline_step('ALS', {'lam': lam, 'p': p}, label)
        except ValueError:
            messagebox.showerror("Error", "Numbers required")

    def _apply_offset(self):
        try:
            val_str = self.offset_target_val_var.get()
            wn_str = self.offset_point_var.get()
            if not wn_str:
                messagebox.showerror("Error", "Please specify a wavenumber.")
                return
            val = float(val_str)
            wn = float(wn_str)
            if self.wavenumbers is not None:
                if wn < np.min(self.wavenumbers) or wn > np.max(self.wavenumbers):
                    messagebox.showwarning("Warning", f"Wavenumber {wn} is outside the current range.")
            label = f"Offset (set {val} at {wn} cm-1)"
            self._add_pipeline_step('Offset', {'val': val, 'wn': wn}, label)
        except ValueError:
            messagebox.showerror("Error", "Invalid numbers provided.")

    def _add_pipeline_step(self, method_type, params=None, label=None):
        step = {
            'type': method_type,
            'params': params if params else {},
            'label': label if label else method_type
        }
        self.pipeline_steps.append(step)
        self._refresh_pipeline_ui()
        self._run_pipeline()

    def _remove_pipeline_step(self, index):
        if 0 <= index < len(self.pipeline_steps):
            del self.pipeline_steps[index]
            self._refresh_pipeline_ui()
            self._run_pipeline()

    def _move_pipeline_step(self, index, direction):
        new_index = index + direction
        if 0 <= new_index < len(self.pipeline_steps):
            self.pipeline_steps[index], self.pipeline_steps[new_index] = \
                self.pipeline_steps[new_index], self.pipeline_steps[index]
            self._refresh_pipeline_ui()
            self._run_pipeline()

    def _run_pipeline(self):
        if self.tensor_data is None: return
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
                elif m_type == 'Offset':
                    if self.wavenumbers is None: raise ValueError("No wavenumbers for offset.")
                    current_data = prep.apply_offset(current_data, self.wavenumbers, p['wn'], p['val'])
                elif m_type == 'SG':
                    current_data = prep.apply_savgol(current_data, p['w'], p['poly'], p['deriv'])
            self.preprocessed_tensor = current_data
            if self.pipeline_steps: self.show_preprocessed_var.set(True)
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Pipeline Error", f"Error in step {m_type}:\n{e}")
            self.preprocessed_tensor = None

    def _refresh_pipeline_ui(self):
        for widget in self.pipeline_scroll.winfo_children(): widget.destroy()
        for i, step in enumerate(self.pipeline_steps):
            card = ctk.CTkFrame(self.pipeline_scroll,
                                fg_color="gray85" if ctk.get_appearance_mode() == "Light" else "gray25")
            card.pack(fill="x", pady=2, padx=2)
            lbl_text = f"{i + 1}. {step['label']}"
            ctk.CTkLabel(card, text=lbl_text, font=("Arial", 11)).pack(side="left", padx=5)
            ctk.CTkButton(card, text="X", width=25, height=25, fg_color="#C0392B",
                          command=lambda idx=i: self._remove_pipeline_step(idx)).pack(side="right", padx=2)
            if i < len(self.pipeline_steps) - 1:
                ctk.CTkButton(card, text="▼", width=25, height=25, fg_color="gray50",
                              command=lambda idx=i: self._move_pipeline_step(idx, 1)).pack(side="right", padx=1)
            if i > 0:
                ctk.CTkButton(card, text="▲", width=25, height=25, fg_color="gray50",
                              command=lambda idx=i: self._move_pipeline_step(idx, -1)).pack(side="right", padx=1)

    def _get_selection_matrix(self):
        if not self.selected_coords:
            messagebox.showerror("Error", "No selection")
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
        except Exception as e:
            messagebox.showerror("PCA Error", f"{e}")

    def _run_mcr_als(self):
        data = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
        if data is None: return
        try:
            n = int(self.mcr_n_components_var.get())
            max_iter = int(self.mcr_max_iter_var.get())
            tol = float(self.mcr_tol_var.get())

            st_fix_indices = []
            if self.mcr_st_fix_var.get():
                st_fix_indices = [int(i) for i in self.mcr_st_fix_var.get().split(',')]

            self.mcr_results = anal.run_mcr_als(
                data, n,
                non_negative=self.mcr_non_negative_var.get(),
                norm=self.mcr_norm_var.get(),
                closure=self.mcr_closure_var.get(),
                unimodal=self.mcr_unimodal_var.get(),
                use_spexfa_init=self.mcr_spexfa_init_var.get(),
                max_iter=max_iter,
                tol_err_change=tol,
                st_init=self.mcr_st_init,
                st_fix_indices=st_fix_indices
            )
            self.current_plot_mode = 'MCR'
            self.last_analysis_mode = 'MCR'
            self._update_pls_target_options()
            self.update_plot()
        except Exception as e:
            messagebox.showerror("MCR Error", f"{e}")

    def _run_parafac(self):
        data = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
        if data is None: return
        try:
            rank = int(self.parafac_rank_var.get())
            self.parafac_results = anal.run_parafac(data, rank, self.parafac_non_negative_var.get())
            self.current_plot_mode = 'PARAFAC'
            self.last_analysis_mode = 'PARAFAC'
            self._update_pls_target_options()
            self.update_plot()
        except Exception as e:
            messagebox.showerror("PARAFAC Error", f"{e}")

    def _run_umap(self):
        X, labels = self._get_selection_matrix()
        if X is None: return
        try:
            emb = anal.run_umap(X)
            self.manifold_results = {'scores': emb, 'labels': labels, 'method_name': 'UMAP'}
            self.current_plot_mode = 'MANIFOLD'
            self.last_analysis_mode = 'MANIFOLD'
            self.update_plot()
        except Exception as e:
            messagebox.showerror("UMAP Error", f"{e}")

    def _run_tsne(self):
        X, labels = self._get_selection_matrix()
        if X is None: return
        try:
            emb = anal.run_tsne(X)
            self.manifold_results = {'scores': emb, 'labels': labels, 'method_name': 't-SNE'}
            self.current_plot_mode = 'MANIFOLD'
            self.last_analysis_mode = 'MANIFOLD'
            self.update_plot()
        except Exception as e:
            messagebox.showerror("t-SNE Error", f"{e}")

    def _run_heatmap(self):
        self.current_plot_mode = 'HEATMAP'
        self.last_analysis_mode = 'HEATMAP'
        self.update_plot()

    def _update_pls_target_options(self):
        self.pls_target_map.clear()
        options = []
        if self.mcr_results:
            k = self.mcr_results['rank']
            for i in range(k):
                name = f"MCR Conc. Comp {i + 1}"
                self.pls_target_map[name] = self.mcr_results['C'][:, i]
                options.append(name)
        if self.parafac_results:
            k = self.parafac_results['factors'][0].shape[1]
            factors = self.parafac_results['factors']
            for i in range(k):
                name_n = f"PARAFAC Trend N Comp {i + 1}"
                trend_n = np.repeat(factors[1][:, i], self.m_cols)
                self.pls_target_map[name_n] = trend_n
                options.append(name_n)
                name_m = f"PARAFAC Trend M Comp {i + 1}"
                trend_m = np.tile(factors[2][:, i], self.n_rows)
                self.pls_target_map[name_m] = trend_m
                options.append(name_m)
        if not options: options = ["No results (Run MCR/PARAFAC)"]
        if hasattr(self, 'pls_target_menu'):
            self.pls_target_menu.configure(values=options)
            self.pls_target_var.set(options[0])

    def _run_pls(self):
        target_name = self.pls_target_var.get()
        if target_name not in self.pls_target_map:
            messagebox.showerror("Error", "Select valid target from list.")
            return
        y = self.pls_target_map[target_name]
        data = self._get_active_data()
        if data is None: return
        try:
            self.pls_results = anal.run_pls(data, y)
            self.current_plot_mode = 'PLS_RESULTS'
            self.last_analysis_mode = 'PLS_RESULTS'
            self.update_plot()
        except Exception as e:
            messagebox.showerror("PLS Error", f"{e}")

    def _run_3dcos(self):
        data = self._get_active_data()
        if data is None: return
        try:
            tensor_no_nan = np.nan_to_num(data, nan=0.0)
            w, n, m = tensor_no_nan.shape
            axis = self.cos_axis_var.get()
            phi_list, psi_list = [], []
            if "Rows" in axis:
                modulator_size = n
                if m < 2: raise ValueError("Not enough columns to correlate rows.")
                for i in range(n):
                    phi, psi = anal.calculate_2dcos(tensor_no_nan[:, i, :])
                    phi_list.append(phi);
                    psi_list.append(psi)
            else:
                modulator_size = m
                if n < 2: raise ValueError("Not enough rows to correlate columns.")
                for j in range(m):
                    phi, psi = anal.calculate_2dcos(tensor_no_nan[:, :, j])
                    phi_list.append(phi);
                    psi_list.append(psi)
            self.cos_3d_results = {
                'phi': np.stack(phi_list, axis=-1),
                'psi': np.stack(psi_list, axis=-1)
            }
            if hasattr(self, 'cos_slider'):
                self.cos_slider.configure(to=modulator_size - 1, number_of_steps=modulator_size - 1)
                self.cos_slider.set(0)
                self.cos_slice_var.set(0)
            self.current_plot_mode = '3DCOS_SLICER'
            self.last_analysis_mode = '3DCOS_SLICER'
            self.update_plot()
        except Exception as e:
            messagebox.showerror("2D-COS Error", f"{e}")

    def _load_mcr_st_init(self):
        if self.wavenumbers is None:
            messagebox.showerror("Error", "No wavenumber axis.")
            return
        paths = filedialog.askopenfilenames(title="Select known spectra files")
        if not paths: return
        loaded_spectra = []
        for p in paths:
            try:
                wav, absorb = data_io.load_single_csv(p)
                if not np.array_equal(self.wavenumbers, wav):
                    raise ValueError("Wavenumber mismatch")
                loaded_spectra.append(absorb)
            except Exception as e:
                messagebox.showerror("Error", f"File {p}: {e}")
                return
        self.mcr_st_init = np.array(loaded_spectra)
        indices_str = ",".join(map(str, range(len(paths))))
        self.mcr_st_fix_var.set(indices_str)
        messagebox.showinfo("Info", f"Loaded {len(paths)} spectra. Suggested indices: {indices_str}")

    def _on_cos_slider(self, val):
        self.cos_slice_var.set(int(val))
        self.update_plot()

    def _run_pca_reconstruction(self):
        if self.pca_results is None:
            messagebox.showerror("Error", "Run PCA first")
            return
        try:
            k = int(self.pca_recon_components_var.get())
            self.pca_results.update(anal.run_pca_reconstruction(self.pca_results, k))
            self.current_plot_mode = 'RECONSTRUCTION'
            self.last_analysis_mode = 'RECONSTRUCTION'
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"{e}")

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
        except Exception as e:
            messagebox.showerror("Tucker Error", f"{e}")

    def _run_fa_rank_analysis(self):
        X, labels = self._get_selection_matrix()
        if X is None: return
        try:
            self.fa_results = anal.run_fa_rank_analysis(X)
            self.fa_results['labels'] = labels
            self.current_plot_mode = 'FA_RANK_RESULTS'
            self.last_analysis_mode = 'FA_RANK_RESULTS'
            self.update_plot()
        except Exception as e:
            messagebox.showerror("FA Error", f"{e}")

    def _run_fa_reconstruction(self):
        if self.fa_results is None:
            messagebox.showerror("Error", "Run Rank Analysis first.")
            return
        try:
            k = int(self.fa_recon_components_var.get())
            recon_res = anal.run_fa_reconstruction(self.fa_results, k)
            self.fa_results.update(recon_res)
            self.fa_results['recon_k'] = k
            self.current_plot_mode = 'FA_RECON_RESULTS'
            self.last_analysis_mode = 'FA_RECON_RESULTS'
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"{e}")

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
        except Exception as e:
            messagebox.showerror("SPEXFA Error", f"{e}")

    def _run_tensor_recon(self, method):
        data = self._get_active_data()
        if data is None: return
        try:
            k = int(self.tensor_recon_components_var.get())
            if method == 'parafac':
                if self.parafac_results is None:
                    messagebox.showerror("Error", "Run PARAFAC analysis first.")
                    return
                res = anal.run_parafac_reconstruction(self.parafac_results, data, k)
                self.tensor_recon_results = {**res, 'source': 'parafac', 'k': k}
            elif method == 'mcr':
                if self.mcr_results is None:
                    messagebox.showerror("Error", "Run MCR-ALS analysis first.")
                    return
                res = anal.run_mcr_reconstruction(self.mcr_results, data, k)
                self.tensor_recon_results = {**res, 'source': 'mcr', 'k': k}
            self.current_plot_mode = 'TENSOR_RECONSTRUCTION'
            self.last_analysis_mode = 'TENSOR_RECONSTRUCTION'
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def _create_plot_canvas(self):
        self.fig = matplotlib.figure.Figure(figsize=(10, 7), dpi=100)
        self.axes = self.fig.subplots(2, 2)
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.toolbar = matplotlib.backends.backend_tkagg.NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.mpl_connect('button_press_event', self._on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self._on_canvas_release)

    def update_plot(self):
        self.zoom_rects.clear()
        self.zoom_start.clear()
        self.initial_lims.clear()
        self.fig.clear()
        mode = self.current_plot_mode

        if mode in ['SPECTRA', 'HEATMAP', 'PLS_RESULTS', 'MANIFOLD']:
            ax = self.fig.add_subplot(111)
            if mode == 'SPECTRA':
                data = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
                suffix = "(Preprocess)" if self.show_preprocessed_var.get() else "(Raw)"
                vis.plot_spectra(ax, self.wavenumbers, data, self.selected_coords, suffix)
            elif mode == 'HEATMAP':
                data = self.preprocessed_tensor if self.show_preprocessed_var.get() and self.preprocessed_tensor is not None else self.tensor_data
                vis.plot_heatmap(ax, data, self.wavenumbers, self.n_rows, self.m_cols)
            elif mode == 'PLS_RESULTS' and self.pls_results:
                vis.plot_pls(ax, self.pls_results, self.wavenumbers)
            elif mode == 'MANIFOLD' and self.manifold_results:
                vis.plot_manifold(ax, self.manifold_results)
        else:
            self.axes = self.fig.subplots(2, 2)
            if mode == 'PCA' and self.pca_results:
                vis.plot_pca(self.axes, self.pca_results, self.wavenumbers)
            elif mode == 'MCR' and self.mcr_results:
                vis.plot_mcr(self.axes, self.mcr_results, self.wavenumbers, self.n_rows, self.m_cols)
            elif mode == 'PARAFAC' and self.parafac_results:
                factors = self.parafac_results['factors']
                ax1, ax2, ax3, ax4 = self.axes.flat
                for ax_temp in self.axes.flat: ax_temp.set_visible(True); vis.apply_theme(ax_temp)
                k = factors[0].shape[1]
                if self.wavenumbers is not None: ax1.plot(self.wavenumbers,
                                                          factors[0]); vis.invert_xaxis_if_wavenumbers(ax1,
                                                                                                       self.wavenumbers)
                ax1.set_title("PARAFAC: Spectra")
                ax2.plot(factors[1], 'o-');
                ax2.set_title("Mode 1")
                ax3.plot(factors[2], 'o-');
                ax3.set_title("Mode 2")
                ax4.bar(range(k), self.parafac_results['weights']);
                ax4.set_title("Weights")
            elif mode == 'TUCKER_RESULTS' and self.tucker_results:
                vis.plot_tucker(self.axes, self.tucker_results, self.wavenumbers, self.n_rows, self.m_cols)
            elif mode == 'FA_RANK_RESULTS' and self.fa_results:
                vis.plot_fa_rank(self.axes, self.fa_results)
            elif mode == 'SPEXFA_RESULTS' and self.spexfa_results:
                vis.plot_spexfa(self.axes, self.spexfa_results, self.wavenumbers)
            elif mode in ['RECONSTRUCTION', 'FA_RECON_RESULTS', 'TENSOR_RECONSTRUCTION']:
                res = None
                if mode == 'RECONSTRUCTION':
                    res = self.pca_results
                elif mode == 'FA_RECON_RESULTS':
                    res = self.fa_results
                elif mode == 'TENSOR_RECONSTRUCTION':
                    res = self.tensor_recon_results
                if res:
                    if 'source_tensor' not in res and 'X_original' not in res: res[
                        'source_tensor'] = self._get_active_data()
                    vis.plot_reconstruction(self.axes, res, self.wavenumbers, self.selected_coords)
            elif mode == '3DCOS_SLICER' and self.cos_3d_results:
                if hasattr(self, 'cos_slider_frame'): self.cos_slider_frame.pack(fill="x", pady=2)
                slice_idx = int(self.cos_slice_var.get())
                vis.plot_2dcos(self.axes, self.cos_3d_results, self.wavenumbers, slice_idx)

        if mode != '3DCOS_SLICER' and hasattr(self, 'cos_slider_frame'):
            self.cos_slider_frame.pack_forget()

        self.canvas.draw()
        for ax in self.fig.axes:
            if ax.get_visible():
                self.initial_lims[ax] = (ax.get_xlim(), ax.get_ylim())
        self.focus_set()

    def _on_canvas_press(self, event):
        self.focus_set()
        ax = event.inaxes
        if ax is None: return
        if event.button == 1:
            self.zoom_start[ax] = (event.xdata, event.ydata)
            rect = Rectangle((event.xdata, event.ydata), 0, 0, fill=True, facecolor='red', alpha=0.2, edgecolor='red',
                             linestyle='-', linewidth=1)
            self.zoom_rects[ax] = rect
            ax.add_patch(rect)
            self.canvas.draw()
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
        if event.button != 1: return
        active_ax = None
        for ax in self.zoom_start:
            if ax.figure == self.fig:
                active_ax = ax
                break
        if active_ax and active_ax in self.zoom_rects:
            rect = self.zoom_rects[active_ax]
            x0, y0 = self.zoom_start[active_ax]
            x1 = event.xdata
            y1 = event.ydata
            if x1 is not None and y1 is not None:
                rect.set_width(x1 - x0)
                rect.set_height(y1 - y0)
                rect.set_xy((x0, y0))
                self.canvas.draw_idle()

    def _on_canvas_release(self, event):
        if event.button == 1:
            active_ax = next(iter(self.zoom_start), None)
            if active_ax and active_ax in self.zoom_rects:
                rect = self.zoom_rects[active_ax]
                rect.remove()
                del self.zoom_rects[active_ax]
                x0, y0 = self.zoom_start[active_ax]
                del self.zoom_start[active_ax]
                x1, y1 = event.xdata, event.ydata
                if x1 is not None and y1 is not None:
                    if abs(x1 - x0) > 1e-5 or abs(y1 - y0) > 1e-5:
                        cur_xlim = active_ax.get_xlim()
                        if cur_xlim[0] > cur_xlim[1]:
                            active_ax.set_xlim(max(x0, x1), min(x0, x1))
                        else:
                            active_ax.set_xlim(min(x0, x1), max(x0, x1))
                        active_ax.set_ylim(min(y0, y1), max(y0, y1))
                self.canvas.draw()
            self.zoom_rects.clear()
            self.zoom_start.clear()

    def _save_project(self):
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if f:
            try:
                status_map = {f"{r},{c}": s for (r, c), s in self.field_matrix_status.items()}
                data_io.save_project_json(f, self.n_rows, self.m_cols, self.original_wavenumbers,
                                          self.original_tensor_data, status_map, self.pipeline_steps)
                messagebox.showinfo("Success", "Project saved.")
            except Exception as e:
                messagebox.showerror("Save Error", f"Cannot save project:\n{e}")

    def _load_project(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if f:
            try:
                d = data_io.load_project_json(f)
                self.n_var.set(d['n_rows'])
                self.m_var.set(d['m_cols'])
                self._create_field_matrix()
                self.original_wavenumbers = d['original_wavenumbers']
                self.original_tensor_data = d['original_tensor_data']
                self._reset_wavenumber_range()
                loaded_status = d['field_matrix_status']
                for str_coords, status in loaded_status.items():
                    r, c = map(int, str_coords.split(','))
                    coords = (r, c)
                    self._update_cell_status(coords, status, f"{coords}\n{status}")
                saved_pipeline = d.get('pipeline_steps', [])
                if saved_pipeline:
                    self.pipeline_steps = saved_pipeline
                    self._refresh_pipeline_ui()
                    self._run_pipeline()
                self.update_plot()
                messagebox.showinfo("Success", "Project loaded.")
            except Exception as e:
                messagebox.showerror("Load Error", f"Cannot load project:\n{e}")
                self._create_field_matrix()

    def _export_to_xlsx(self):
        f = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if f:
            try:
                results = {
                    'pca': getattr(self, 'pca_results', None),
                    'mcr': getattr(self, 'mcr_results', None),
                    'parafac': getattr(self, 'parafac_results', None),
                    'tucker': getattr(self, 'tucker_results', None),
                    'spexfa': getattr(self, 'spexfa_results', None),
                    'fa': getattr(self, 'fa_results', None),
                    'pls': getattr(self, 'pls_results', None),
                    'manifold': getattr(self, 'manifold_results', None),
                    'recon': getattr(self, 'tensor_recon_results', None) if getattr(self, 'tensor_recon_results',
                                                                                    None) else
                    (getattr(self, 'pca_results', None) if self.current_plot_mode == 'RECONSTRUCTION' else None)
                }
                data_io.export_results_to_xlsx(f, self.wavenumbers, self.tensor_data, self.preprocessed_tensor,
                                               self.n_rows, self.m_cols, results, getattr(self, 'pipeline_steps', None))
                messagebox.showinfo("Success", "Export successful.")
            except Exception as e:
                print(f"DETAILED ERROR: {e}")
                messagebox.showerror("Export Error", f"Cannot export data:\n{e}")

    def _reset_analysis_results(self):
        self.pca_results = None
        self.mcr_results = None
        self.parafac_results = None
        self.manifold_results = None

    def _on_global_click(self, event):
        if isinstance(event.widget, tkinter.Entry): return
        self.focus_set()


if __name__ == "__main__":
    app = ChemTensorApp()
    app.mainloop()