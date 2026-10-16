# Cuttlestrike — stratégies de chasse de la seiche

Analyse du comportement de chasse de la seiche (*Sepia*) à partir du tracking vidéo (DeepLabCut) :
features par frame → états comportementaux (clustering Leiden) → séquences d'états par individu
→ stratégies de chasse (distance SeqVec + groupes Leiden kNN).

## Organisation

```
Code/
  all_cuttlefish.ipynb          notebook principal (français)
  all_cuttlefish_english.ipynb  même notebook en anglais
  add_features_full.py          calcule les features (SOURCE="cut" : début -> frappe)  -> Data/cut_cuttlefish_with_new_features_smooth_*.csv
  add_features_cut_2.py         ancien calcul des features (version de juillet)        -> Data/cut_cuttlefish_with_features_smooth_*.csv
Data/
  cut_combined_cuttlefish_tracking.csv           tracking DeepLabCut coupé (début -> frappe), entrée de add_features_*
  cut_cuttlefish_with_new_features_smooth_1.csv  features utilisées par le notebook (DATASET = "cut_new")
  cut_cuttlefish_with_features_smooth_1.csv      anciennes features (DATASET = "cut_old")
  Data_cuttlfish_strike.xlsx                     informations par individu (frames de début / frappe, échelle, fps, type de capture)
  Data_cuttlfish_strike_complet.xlsx             version complète (analyses comparatives)
  Classeur1.xlsx                                 données complémentaires (analyse des touches)
  mapping_optimal_shrimp.json                    stratégies de la crevette (comparaison seiche x crevette)
Results/
  reference_best/               session de référence (voir ci-dessous)
Figures/                        figures produites par le notebook
```

## Lancer l'analyse

```bash
pip install -r requirements.txt
cd Code
jupyter notebook all_cuttlefish.ipynb
```

Les chemins sont **relatifs** : le notebook doit être ouvert depuis `Code/`.
Exécuter les cellules dans l'ordre : configuration (cellule 0) → clustering Leiden des frames (cellule 2)
→ explorateur SeqVec → analyses, vidéos, explication des groupes.

Les vidéos ne sont pas incluses (trop lourdes) : les placer dans `Videos/videos_catch/`
(ou changer `VIDEO_ROOT` dans la cellule 0) pour utiliser le viewer et le navigateur vidéo.

## Recharger la session de référence

`Results/reference_best/` contient tout l'état de l'analyse : configuration, clusters et UMAP,
séquences, distance SeqVec, groupes de stratégies, tableaux d'explication des groupes, figures
et copie du code qui l'a produite.

Dans la dernière cellule du notebook :
```python
DO_LOAD   = True
LOAD_NAME = "reference_best"
```
puis exécuter la cellule (après la cellule 0, sans relancer la cellule 2).

## Régénérer les données

`Code/add_features_full.py` recalcule les features à partir de `Data/cut_combined_cuttlefish_tracking.csv`
et de `Data/Data_cuttlfish_strike.xlsx`. Les fichiers de lissage 10 et 30, ainsi que la version complète
(`SOURCE = "full"`, tracking `combined_cuttlefish_tracking.csv` de 662 Mo), dépassent la limite de GitHub
et ne sont pas inclus : ils se régénèrent avec ce script.
