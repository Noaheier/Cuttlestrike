# Cuttlestrike — cuttlefish hunting strategies

Analysis of cuttlefish (*Sepia*) hunting behaviour from video tracking (DeepLabCut):
per-frame features → behavioural states (Leiden clustering) → per-individual state sequences
→ hunting strategies (SeqVec distance + Leiden kNN groups).

## Layout

```
Code/
  1_compute.ipynb               full analysis WITHOUT figures (clustering, strategies, regrouping) + session save
  2_model.ipynb                 model characterisation (UMAP, frame types, medoids, probabilities, group explanation)
  3_results.ipynb               results figures (variables, comparisons, significance, trajectories, polar / KDR, catch types)
  4_interactive.ipynb           interactive tools (SeqVec explorer, 3D, video viewer / browser, simulator)
  all_cuttlefish.ipynb          complete notebook (French)
  all_cuttlefish_english.ipynb  same notebook in English
  add_features_full.py          computes the features (SOURCE="cut": start -> strike)  -> Data/cut_cuttlefish_with_new_features_smooth_*.csv
  add_features_cut_2.py         previous feature computation (July version)             -> Data/cut_cuttlefish_with_features_smooth_*.csv
Data/
  cut_combined_cuttlefish_tracking.csv           cut DeepLabCut tracking (start -> strike), input of add_features_*
  cut_cuttlefish_with_new_features_smooth_1.csv  features used by the notebook (DATASET = "cut_new")
  cut_cuttlefish_with_features_smooth_1.csv      previous features (DATASET = "cut_old")
  Data_cuttlfish_strike.xlsx                     per-individual information (start / strike frames, scale, fps, catch type)
  Data_cuttlfish_strike_complet.xlsx             complete version (comparative analyses)
  Classeur1.xlsx                                 additional data (hit analysis)
  mapping_optimal_shrimp.json                    shrimp strategies (cuttlefish x shrimp comparison)
Results/
  reference_best/               reference session (see below)
Figures/                        figures produced by the notebook
```

## Running the analysis

**Split notebooks (recommended)** — open them from `Code/`:
1. `1_compute.ipynb`: run everything, then set `DO_SAVE = True` and a `SAVE_NAME` in the save cell and rerun it.
2. `2_model.ipynb`, `3_results.ipynb`, `4_interactive.ipynb`: set `LOAD_NAME` to that session
   (default `"reference_best"`) and run only the figures you need — nothing is recomputed.

**Complete notebook** — `all_cuttlefish_english.ipynb` (or the French version) contains everything in one file:

```bash
pip install -r requirements.txt
cd Code
jupyter notebook all_cuttlefish_english.ipynb
```

Paths are **relative**: the notebook must be opened from `Code/`.
Run the cells in order: configuration (cell 0) → Leiden clustering of the frames (cell 2)
→ SeqVec explorer → analyses, videos, group explanation.

Videos are not included (too large): put them in `Videos/videos_catch/`
(or change `VIDEO_ROOT` in cell 0) to use the video viewer and the strategy browser.

## Reloading the reference session

`Results/reference_best/` contains the full state of the analysis: configuration, clusters and UMAP,
sequences, SeqVec distance, strategy groups, group explanation tables, figures
and a copy of the code that produced it.

In the last cell of the notebook:
```python
DO_LOAD   = True
LOAD_NAME = "reference_best"
```
then run the cell (after cell 0, without rerunning cell 2).

## Regenerating the data

`Code/add_features_full.py` recomputes the features from `Data/cut_combined_cuttlefish_tracking.csv`
and `Data/Data_cuttlfish_strike.xlsx`. The smoothing 10 and 30 files, as well as the full version
(`SOURCE = "full"`, 662 MB `combined_cuttlefish_tracking.csv` tracking), exceed GitHub's file size limit
and are not included: they can be regenerated with this script.
