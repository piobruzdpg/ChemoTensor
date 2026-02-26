# Plan Refaktoryzacji: ChemTensor Explorer

## Cel
Rozbicie monolitycznego pliku `main.py` na niezależne, łatwe w utrzymaniu moduły. Oddzielenie warstwy wizualnej (GUI) od warstwy zarządzania stanem danych (macierze widm, wyniki analiz) oraz od logiki wykresów.

## Faza 1: Porządki z narzędziami pomocniczymi (Utilities)
- [ ] Utworzyć plik `modules/ui_utils.py` (lub `widgets.py`).
- [ ] Przenieść klasę `ToolTip` z `main.py` do nowego pliku.
- [ ] Przenieść klasę `FileManagerWindow` z `main.py`.
*Dlaczego?* Te klasy są uniwersalne i całkowicie niezależne od głównej logiki programu. Zaśmiecają główny plik.

## Faza 2: Zarządzanie Stanem i Danymi (Data Model)
- [ ] Utworzyć plik `modules/state_manager.py` (lub `data_model.py`).
- [ ] Stworzyć w nim klasę `SessionState`, która przejmie wszystkie zmienne przechowujące dane:
    - `original_tensor_data`, `tensor_data`, `preprocessed_tensor`
    - `wavenumbers`, `original_wavenumbers`
    - `pipeline_steps`
    - Słowniki z wynikami: `pca_results`, `mcr_results`, `parafac_results` itd.
    - Metody takie jak `get_active_data()`, `get_selection_matrix()` czy logika resetowania zakresu liczb falowych.
*Dlaczego?* Dzięki temu, jeśli w przyszłości będziesz chciał dodać funkcję analizy w tle albo zautomatyzować skrypt bez włączania okienek, logika danych pozostanie nienaruszona.

## Faza 3: Menedżer Wykresów (Plot Controller)
- [ ] Utworzyć plik `modules/plot_manager.py`.
- [ ] Utworzyć klasę `InteractivePlotManager`, która w swoim `__init__` przyjmie ramkę (`plot_frame`), w której ma się narysować.
- [ ] Przenieść z `main.py` metody:
    - `_create_plot_canvas()`
    - Obsługę zdarzeń Matplotlib: `_on_canvas_press`, `_on_canvas_motion`, `_on_canvas_release` (zoomowanie).
    - Metodę `update_plot()`.
*Dlaczego?* Tkinter i Matplotlib to dwa różne światy. `main.py` nie powinien zajmować się obliczaniem współrzędnych kursora (zoom_rects) ani czyszczeniem osi.

## Faza 4: Rozbicie Głównego Interfejsu (UI Modules)
- [ ] Rozbić gigantyczną metodę `_create_control_widgets` na osobne klasy widoków.
- [ ] Utworzyć plik `modules/ui_tabs.py`.
- [ ] Stworzyć osobne klasy dla zakładek dziedziczące po `ctk.CTkFrame`:
    - `DataTabView` (zawiaduje siatką `field_matrix`, wczytywaniem i zapisem).
    - `PreprocessTabView` (zawiaduje suwakami i przyciskami do SNV, ALS, itp.).
    - `AnalysisTabView` (zawiaduje przyciskami do uruchamiania PCA, MCR, SPEXFA).
*Dlaczego?* Interfejs staje się modułowy. Jeśli zechcesz dodać zakładkę "Klasyfikacja (SVM / Random Forest)", po prostu utworzysz nową klasę widoku, zamiast doklejać kolejne 300 linijek do `main.py`.

## Faza 5: Nowy, odchudzony `main.py`
- [ ] Przebudować klasę `ChemTensorApp` w `main.py` tak, aby pełniła jedynie rolę "dyrygenta" spinającego moduły.
- [ ] Struktura nowego `main.py` powinna sprowadzać się do:
    1. Inicjalizacji `self.state = SessionState()`.
    2. Inicjalizacji `self.plot_manager = InteractivePlotManager(self.plot_frame, self.state)`.
    3. Podpięcia zakładek: `self.data_tab = DataTabView(self.tab_view, self.state, self.plot_manager)`, itd.
- [ ] `main.py` powinien mieć docelowo nie więcej niż 150-200 linijek kodu.

## Opcjonalne: Ujednolicenie Callbacks
- [ ] Wprowadzić system "sygnałów/zdarzeń" lub proste funkcje `callback`, aby zakładki UI mogły informować `PlotManager` o konieczności odświeżenia płótna po np. dodaniu nowego filtru wygładzającego, bez bezpośredniego odwoływania się do głównej aplikacji.