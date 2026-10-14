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

## Datasets

| Dataset | Used for | What it contains | Get it |
|---|---|---|---|
| [Sleep-EDF Database Expanded 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) (PhysioNet) | Part 1, sleep staging | 197 overnight recordings from 100 people: Sleep Cassette (healthy, age 25–101) and Sleep Telemetry (mild sleep difficulty). EEG Fpz-Cz, EEG Pz-Oz and horizontal EOG at 100 Hz, with expert hypnograms (R&K stages). 8.1 GB, open access | [Direct zip](https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip); unzip to `Dataset/archive/sleep-edf-database-expanded-1.0.0/` |
| [PhysioNet Challenge 2026](https://moody-challenge.physionet.org/2026/) (Human Sleep Project) | Part 2, cognitive impairment | Clinical polysomnography (EEG, EOG, chin and leg EMG, ECG, airflow, effort belts, SpO2) from several hospitals, with automatic CAISR sleep-stage annotations, demographics and a later cognitive impairment diagnosis. We used the large training set: 6,522 people from 3 hospitals (about 1.2 TB) | From the Challenge page, under its data terms; not republished here |

The raw recordings are not in this repository. The scripts rebuild everything else from them;
included are the trained network `checkpoints/sleep_stager.pt` (7.7 MB) and the small summary
tables `data/features.csv` and `data/sleep_edf/index.csv`.

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
streams them one night at a time (parallel workers, checkpointed, optionally sharded) and keeps
only a small table of features:

```
python hsp_features.py <path to the Challenge 2026 data> --out hsp_features.csv
```

Put `hsp_features.csv` and the dataset's `demographics.csv` in `data/hsp_large/`, then:

```powershell
python hsp_baseline.py data/hsp_large   # age only, logistic regression, random forest, XGBoost
python hsp_explain.py data/hsp_large    # SHAP explanations, risk groups, worked examples
```

Use the model on a new person:

```powershell
python predict.py train data/hsp_large   # fits the final model on everyone, saves checkpoints/cognitive_model.joblib
python predict.py night <recording.edf> <caisr_annotations.edf> --age 67 --sex female
python predict.py table data/hsp_large/hsp_features.csv <record> --age 67 --sex female   # a night already measured
```

It prints a risk score, how it ranks against everyone and against people of the same age, the
risk group (in cross-validation, 31.7% of the high group were later impaired against 3.0% of the
low group) and the five features that moved the risk most. It is a research model, not a diagnosis.

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
| `predict.py` | Saves the final model and scores one new night |
| `config.py` | Model size and training settings |
| `demo.html`, `design.html` | Project presentation and system design |

All images and reports are written to `results/`.

Training uses full precision: half precision (autocast) produces NaNs on GTX 16-series GPUs.
