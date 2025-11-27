# ChemTensor

**ChemTensor** is a modular, GUI-based application designed for advanced chemometric analysis of spectral data (e.g., FTIR, Raman, UV-Vis). Built with **Python** and **CustomTkinter**, it allows researchers to visualize, preprocess, and model complex datasets using matrix and tensor decomposition methods.

The tool is particularly optimized for analyzing data arranged in grid structures (NxM spatial or temporal measurements), handling everything from basic PCA to multi-way Tensor Analysis (PARAFAC, Tucker).

> **Note:** The user interface is currently in Polish. An English translation of the GUI is planned for upcoming releases.

## Key Features

### 1. Data Management & Visualization
* **Grid-based Data Import:** Load CSV spectral files into a customizable NxM spatial grid.
* **Data Imputation:** Tools to handle missing pixels/samples (interpolation via neighbors, row/col means, or zero-filling).
* **Interactive Visualization:**
    * Spectral plots with zoom/pan capabilities.
    * Heatmaps for spatial distribution analysis.
    * Interactive slicing for 3D/Tensor data.
* **Project I/O:** Save and load full project states via JSON and export comprehensive results to Excel (`.xlsx`).

### 2. Advanced Preprocessing Pipeline
Build a stackable preprocessing pipeline with real-time preview:
* **Smoothing & Derivatives:** Savitzky-Golay filters.
* **Baseline Correction:** Asymmetric Least Squares (ALS).
* **Normalization:** SNV (Standard Normal Variate), Min-Max, L1 Norm, MSC (Multiplicative Scatter Correction).
* **Pipeline Management:** Reorder, add, or remove steps dynamically.

### 3. Chemometric Analysis Modules
ChemTensor Explorer implements a wide range of algorithms for decomposition and regression:

* **Factor Analysis & Dimension Reduction:**
    * **PCA:** Principal Component Analysis (Scores, Loadings, Scree plots).
    * **FA (Malinowski):** Rank analysis using RE (Real Error) and IND functions.
    * **Manifold Learning:** t-SNE and UMAP for non-linear visualization.
    * **SPEXFA:** Spectral Isolation/Extraction.

* **Multivariate Curve Resolution (MCR):**
    * **MCR-ALS:** With support for Non-Negative Least Squares (NNLS) and constraints.
    * **Known Spectra:** Ability to fix known spectra (constraints) during initialization.

* **Tensor Decomposition (Multi-way Analysis):**
    * **PARAFAC:** CP decomposition for trilinear data (Sample x Time/Space x Wavenumber).
    * **Tucker:** Tucker decomposition (Higher-order SVD).
    * **Tensor Reconstruction:** Reconstruct data from factors to analyze residuals.

* **2D Correlation Spectroscopy (2D-COS):**
    * Generate Synchronous and Asynchronous 2D correlation maps.
    * Slicing along the perturbation axis.

* **Regression:**
    * **PLS:** Partial Least Squares regression targeting specific chemical components or PARAFAC/MCR trends.

---

## Installation

### Prerequisites
* Python 3.8+

### Dependencies
Install the required packages using pip and the provided requirements file:

```bash
pip install -r requirements.txt
```

Alternatively, install packages manually:

```bash
pip install numpy pandas scipy matplotlib scikit-learn tensorly pymcr umap-learn openpyxl customtkinter
```

---

## Usage

1.  **Run the application:**
    ```bash
    python main.py
    ```

### Workflow Example

1.  **Tab "Dane" (Data):** Set the grid size (N x M), load your `.csv` files into the cells, and crop the wavenumber range if necessary.
2.  **Tab "Preprocessing":** Add steps like SNV or Savitzky-Golay to clean the spectra. The plot updates automatically.
3.  **Tab "Analiza" (Analysis):** Choose a method (e.g., PCA or PARAFAC).
    * Set the number of components/rank.
    * Click the run button (e.g., "Oblicz PCA").
4.  **Explore:** Check the resulting plots (Scores, Loadings, Heatmaps).
5.  **Export:** Use "Eksportuj do Excela" to save all numerical results, residuals, and preprocessing logs to a spreadsheet.

---

## Project Structure

```text
main.py              # Entry point and GUI implementation (CustomTkinter)
modules/
├── preprocessing.py # Algorithms for signal correction (ALS, SG, MSC)
├── analysis.py      # Core logic for PCA, MCR, PARAFAC, PLS, etc.
├── visualization.py # Matplotlib integration and plotting functions
└── data_io.py       # Handling CSV imports, JSON project state, and Excel exports
```
