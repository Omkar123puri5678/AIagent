# AI-based prediction of cognitive impairment from polysomnography

Final-year engineering project. Can an AI read one night's sleep recording (polysomnography)
and tell who is more likely to develop cognitive impairment later, beyond what age alone tells us?

The project has two parts:

1. **Sleep-stage scoring:** a neural network, trained from scratch, that labels each 30 seconds
   of an overnight recording as W, N1, N2, N3 or REM. Trained on
   [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/).
2. **Cognitive impairment risk:** 49 measurements per night (sleep structure, EEG, heart, breathing,
   oxygen, muscle tone), then models that predict later impairment, with SHAP explanations.
   Trained on the [PhysioNet Challenge 2026](https://moody-challenge.physionet.org/2026/) data
   (Human Sleep Project).

Open **`demo.html`** for the results and **`design.html`** for the block diagram, design flow,
architecture and step-by-step guide.

## Results

| Part 1: sleep staging (20 unseen recordings) | |
|---|---|
| Accuracy | 82.4% |
| Macro F1 | 0.800 |
| Cohen's kappa | 0.767 |

| Part 2: cognitive impairment (6,522 people, 3 hospitals) | Age only | Logistic regression |
|---|---|---|
| Age-conditioned AUROC (5-fold cross-validation × 10) | 0.531 | **0.667** |
| Held-out hospital I0002 | 0.518 | **0.762** |
| Held-out hospital I0006 | 0.533 | **0.690** |
| Held-out hospital S0001 | 0.537 | **0.565** |

The age-conditioned AUROC only compares people within 2 years of age, so it measures what sleep
adds beyond age (0.5 = guessing). The strongest sleep signals are reduced sleep-spindle (sigma)
power, more limb movements, more light N2 sleep and less REM sleep. BMI is left out on purpose:
whether it was recorded at all gave away the label.

## Large files

These are not in this repository:

| Folder | Size | How to get it |
|---|---|---|
| `Dataset/` (Sleep-EDF raw recordings) | 8.1 GB | Download [Sleep-EDF Database Expanded 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) ([direct zip](https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip)) and unzip it to `Dataset/archive/sleep-edf-database-expanded-1.0.0/` |
| `data/sleep_edf/*.npz` (preprocessed epochs) | 4 GB | Run `python preprocess.py` to rebuild it from the raw recordings |
| `data/hsp_large/` (Challenge feature table and demographics) | 3 MB | Patient-level data; get the recordings from the Challenge and run `hsp_features.py` (below). Not republished here |

Included: the trained network `checkpoints/sleep_stager.pt` (7.7 MB), the epoch features
`data/features.csv` and the recording summary `data/sleep_edf/index.csv`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Run

Part 1, sleep staging:

```powershell
python preprocess.py   # raw recordings -> labelled epochs in data/sleep_edf/
python train.py        # trains the model, saves the best one to checkpoints/
python evaluate.py     # scores the saved model on the held-out test subjects
python plots.py        # result images in results/
python explore.py      # one recording: raw signals, filtering, spectrum, time-frequency
python features.py     # band-power and Hjorth features for every epoch
python baseline.py     # logistic regression baseline on those features
```

Part 2, cognitive impairment. The Challenge recordings are about 1.2 TB, so feature extraction
runs on a cloud machine where the data is stored and leaves a small table behind:

```
python hsp_features.py <path to the Challenge 2026 data> --out hsp_features.csv
```

Put `hsp_features.csv` and the dataset's `demographics.csv` in `data/hsp_large/`, then:

```powershell
python hsp_baseline.py data/hsp_large   # age only, logistic regression, random forest, XGBoost
python hsp_explain.py data/hsp_large    # SHAP explanations, risk groups, worked examples
```

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | Filters, epochs, labels and normalises the Sleep-EDF recordings |
| `dataset.py` | Subject-wise train/val/test split and training batches |
| `model.py` | The network: a CNN per epoch, then a bidirectional GRU across epochs |
| `train.py` | Training loop, early stopping, final test report |
| `evaluate.py` | Accuracy, macro F1, Cohen's kappa and the confusion matrix |
| `plots.py` | Result images |
| `explore.py` | Step-by-step look at one raw recording |
| `features.py`, `baseline.py` | Hand-made epoch features and logistic regression baseline |
| `hsp_features.py` | 49 measurements per night of the Challenge 2026 data |
| `hsp_baseline.py` | Cognitive impairment models, cross-validation, leave-one-hospital-out |
| `hsp_explain.py` | SHAP explanations and risk groups |
| `config.py` | Model size and training settings |
| `demo.html`, `design.html` | Project presentation and system design |

All images and reports are written to `results/`.

Training uses full precision: half precision (autocast) produces NaNs on GTX 16-series GPUs.
