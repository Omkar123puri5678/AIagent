# Sleep-stage classifier

A neural network, trained from scratch, that labels each 30 seconds of an overnight
recording as one of five sleep stages (W, N1, N2, N3, REM). It learns from the
[Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) dataset in `Dataset/`.

## Large files

Two folders are too large for GitHub and are not in this repository:

| Folder | Size | How to get it |
|---|---|---|
| `Dataset/` (raw recordings) | 8.1 GB | Download [Sleep-EDF Database Expanded 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) from PhysioNet ([direct zip](https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip)) and unzip it to `Dataset/archive/sleep-edf-database-expanded-1.0.0/` |
| `data/sleep_edf/*.npz` (preprocessed epochs) | 4 GB | Not hosted anywhere; run `python preprocess.py` to rebuild it from the raw recordings (a few minutes) |

Everything else is included, so you can use the trained model without the dataset:

| File | Size | What it is |
|---|---|---|
| `checkpoints/sleep_stager.pt` | 7.7 MB | The trained network |
| `data/features.csv` | 46 MB | Band-power and Hjorth features for all 237,936 epochs |
| `data/sleep_edf/index.csv` | 10 KB | Per-recording summary: subject, night, epochs per stage |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Run

```powershell
python preprocess.py   # raw recordings -> labelled epochs in data/sleep_edf/ (a few minutes)
python train.py        # trains the model, saves the best one to checkpoints/
python evaluate.py     # scores the saved model on the held-out test subjects
python plots.py        # redraws the images in results/ (train.py also does this when it finishes)
```

Signal analysis and a classical baseline (the steps from the Colab notebook):

```powershell
python explore.py      # one recording: raw signals, filtering, re-referencing, spectrum, time-frequency
python features.py     # band-power and Hjorth features for every epoch -> data/features.csv
python baseline.py     # logistic regression on those features, scored on the same test subjects
```

`python explore.py ST7011J` explores a different recording; output goes to `results/explore/<recording>/`.

## Results

Everything lands in `results/`:

| File | Shows |
|---|---|
| `training_curves.png` | Training loss and validation scores per epoch |
| `confusion_matrix.png` | How each true stage was scored on the test set |
| `hypnogram_<recording>.png` | One test night: expert scoring against the model's |
| `test_report.txt` | Test accuracy, macro F1, kappa and per-stage F1 |
| `history.csv` | The per-epoch numbers behind the training curves |

`python plots.py SC4131E` draws the hypnogram for a different test recording.

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | Filters, epochs, labels and normalises the raw recordings |
| `dataset.py` | Subject-wise train/val/test split and training batches |
| `model.py` | The network: a CNN per epoch, then a GRU across epochs |
| `train.py` | Training loop, early stopping, final test report |
| `evaluate.py` | Accuracy, macro F1, Cohen's kappa and the confusion matrix |
| `plots.py` | Result images |
| `explore.py` | Step-by-step look at one raw recording |
| `features.py` | Band-power and Hjorth features per epoch |
| `baseline.py` | Logistic regression baseline on those features |
| `config.py` | Model size and training settings |

## How it works

- **Input:** two EEG channels (Fpz-Cz, Pz-Oz) and one EOG channel at 100 Hz.
- **Model:** a 1D CNN turns each 30-second epoch into a feature vector; a bidirectional
  GRU then reads 20 consecutive epochs, so each one is scored in context.
- **Split:** by subject (80% train, 10% validation, 10% test), so the test score reflects
  people the model has never seen.
- **Imbalance:** rare stages (N1, N3) get a higher weight in the loss.
- **Model selection:** the checkpoint with the best validation macro F1 is kept.

Training uses full precision: half precision (autocast) produces NaNs on GTX 16-series GPUs.
