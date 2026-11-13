# CRD-DS bearing fault diagnosis

This repository implements vibration–acoustic emission bearing diagnosis with adaptive probability calibration, sample-dependent reliability estimation, and reliability-discounted Dempster–Shafer fusion. It also includes the paper's machine-learning and deep-learning comparisons, component ablations, CWRU validation, and DC-offset controls.

The experiments run directly in Python, including PyCharm. Each experiment uses a JSON configuration and a short training entry point. Shared numerical routines are under `common/`.

## Start with the method

```text
Vibration / AE records
  -> chronological split -> noise and window preprocessing
  -> separate CNN branches -> class probabilities
  -> adaptive calibration -> sample-dependent reliability
  -> discounted D-S fusion -> prediction
```

| Read | What it implements |
| --- | --- |
| [Main_Method/CRD_DS/neural_fusion.py](Main_Method/CRD_DS/neural_fusion.py) | Adaptive temperature and bias, reliability MLP, discounted fusion, and fusion loss |
| [Main_Method/CRD_DS/experiment.py](Main_Method/CRD_DS/experiment.py) | Vibration/AE inputs, operating-condition context, and the full experiment |
| [common/primary/branches.py](common/primary/branches.py) | CNN branches, window preparation, and branch training |
| [common/primary/partition.py](common/primary/partition.py) | Physical calibration holdout and overlap removal |
| [common/primary/fitting.py](common/primary/fitting.py) | Fusion-head training, calibration-only epoch selection, and reliability refinement |
| [common/primary/evaluation.py](common/primary/evaluation.py) | Frozen-model evaluation, eight methods, and four-component subset |
| `Deep_Learning/<method>/models.py` | DE-1D-CNN, Attention-1DCNN, and ECA-MSCNN architectures |
| [Machine_Learning/ML_Baselines/baselines.py](Machine_Learning/ML_Baselines/baselines.py) | EMD/WPD features with ANN, DNN, and SVM |

The main model uses a calibration hidden width of 8 and a reliability hidden width of 64. The DC controls use matched eight-unit fusion heads without reliability refinement. CWRU uses a ten-class model with two vibration inputs (DE and FE).

## Environment

Use Python 3.12 and install the dependencies in the environment selected by your IDE:

```bash
python -m pip install -r requirements.txt
```

`device: "auto"` uses CUDA when available and otherwise uses the CPU. The scripts also accept `--device cpu` or `--device cuda`. Full experiment grids can take substantial time on the CPU.

## Data inputs

Set `data_root` in the selected `experiment.json`. Relative paths in a configuration are resolved against that configuration's directory, so execution does not depend on the IDE working directory. Use raw, finite numeric records; splitting, DC treatment, noise injection, and normalization are performed by the code.

The laboratory experiments read paired CSV tables with one column per bearing class (healthy, inner-race, outer-race, rolling-element). Include column headers; do not include row indices or time columns. Class columns must correspond across the two modalities. Matching headers `normal`, `if`, `of`, and `bf` are recognized and aligned automatically; unrecognized headers require the same class order in both tables. Preserve chronological sample order and do not normalize the CSV records before loading them.

```text
Code/data/Laboratory_Dataset/
  Data_Vibration/800/CH_06/data_Vibration_1k_800_CH_06_4c.csv
  Data_AE/800/data_AE_2000k_800_4c.csv
  ... corresponding files for the other configured RPM values
```

CWRU uses two prepared, aligned ten-class CSV files. The experiment data contain 119,808 samples per column; this is the prepared dataset length, not a hard-coded input limit:

```text
Code/data/Data_CWRU_Corrected/
  data_12k_1797_10c_DE.csv
  data_12k_1797_10c_FE.csv
```

Use the same ten-class column order in both CWRU tables, with distinct headers such as `class_0` through `class_9`. The laboratory header aliases above represent four fault types and should not be reused for multiple CWRU fault sizes.

Use the same ten-class column order in both CWRU tables, with distinct headers such as `class_0` through `class_9`. The laboratory header aliases above represent four fault types and should not be reused for multiple CWRU fault sizes.

The laboratory recordings are supplied as MATLAB files. Convert the raw records to the CSV layout above before training, preserving class correspondence and sample order. A MATLAB-to-CSV converter, datasets, trained weights, and generated results are not included in this directory.

## Run in PyCharm

Open the `Code` directory as the project, select its Python environment, and open [train.py](train.py). At the bottom, choose `EXPERIMENT`, then run the file:

```python
EXPERIMENT = "crd_ds"
CONFIG = None       # Use this experiment's experiment.json.
OUTPUT = None       # Use its configured Results directory.
DEVICE = "auto"
DRY_RUN = False
```

Set `DRY_RUN = True` to inspect the selected configuration without loading data or training. Each experiment's own `train.py` can also be run directly in the IDE.

## Short command-line examples

```bash
python train.py --experiment crd_ds --dry-run
python train.py --experiment crd_ds
python train.py --experiment attention
python train.py --experiment ml
```

| Experiment key | Implementation |
| --- | --- |
| `crd_ds` | Main method and component/fusion comparisons |
| `de` | DE-1D-CNN with the supplied frozen DE hyperparameters |
| `attention` | Attention-1DCNN |
| `eca` | ECA-MSCNN |
| `ml` | Six EMD/WPD × ANN/DNN/SVM comparisons |
| `cwru` | CWRU DE/FE validation |
| `dc_removed` | DC-removed reference experiment |
| `dc_retained` | DC-retained experiment, window-mean baseline, and fixed-model controls |

For the DC comparison, run `dc_removed` first, then `dc_retained` with matching RPM, SNR and seed selections. The latter's `reference_results_root` must point to the DC-removed output. Each run requires an empty output directory; use `--output /path/to/new_results` for a separate run.

## Experimental settings

All eight configurations select **−12 through 0 dB in 1 dB steps, plus Clean** (`null` in JSON): 14 conditions. Laboratory experiments use 11 RPM values and ten seeds; CWRU uses 1797 RPM. Each `experiment.json` defines the data paths, selected grid and configurable training parameters. Fixed architecture and selection rules are implemented in the linked source modules.

- Records are split before windowing: 60% training, 7.5% validation, 7.5% calibration, and 25% test.
- Laboratory vibration windows use 512 samples and stride 51; AE windows use 16,384 samples without overlap. Noise is added to continuous splits before windowing.
- The main method trains the CNN branches separately. Fusion heads use a physical calibration Fit/Selection partition with overlapping windows purged. Selection targets 20% of the retained calibration observations, using a common per-RPM quota. Heads use full-batch AdamW with uniform fitting weights; epochs are selected by accuracy balanced across RPM, seed and non-Clean SNR, with the earliest epoch winning ties and no refitting on Selection. Test observations are used only for evaluation.
- A CWRU grid containing only Clean uses Clean for fusion-head selection. The main laboratory method and DC controls require at least one noisy SNR for selection.
- Main-method and CWRU reliability refinement freezes the Full head's calibrators, includes the selected base model as epoch zero, and independently refines the Reliability-only head. The Calibration-only head keeps its selected base model. Widths, learning rates, regularization and epoch counts are in `neural_modules`.
- The deep comparisons use batch size 64 and 100 epochs without early stopping, selecting the earliest maximum validation accuracy. Learning rates and StepLR schedules are defined in each `models.py`. DE hyperparameters are loaded from `Deep_Learning/DE-1D-CNN/Frozen_Parameters/DE_1D_CNN_MY_DATA_Parameters.json`; no DE search is rerun.
- ML comparisons use vibration EMD/WPD features, training-only feature scaling, and ANN/DNN final-epoch models. The feature extraction, classifiers and their hyperparameters are in `baselines.py`. Vibration windows are selected using the same paired-window indices as the two-modality comparisons.
- DC-retained processing intentionally preserves the AE window mean through standard-deviation-only scaling. Do not replace it with the main method's zero-mean window normalization.
- The window-mean control uses GaussianNB with default class priors and the raw noisy AE window mean; its variance smoothing and fixed-model comparison tolerances are configured in the DC-retained JSON.

Outputs are limited to model parameters, essential probabilities, training/selection histories, and metric tables. The main method includes the eight-method table and its four-component subset. All results are written under the selected experiment's `output_root`.
