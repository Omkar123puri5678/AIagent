"""Train the sleep-stage model on the preprocessed data:  python train.py"""

import csv
import math
import sys
import time

import torch
import torch.nn.functional as F

import config
import plots
from dataset import BatchSampler, class_weights, load_recordings, split_recordings
from evaluate import confusion_matrix, metrics, report
from model import SleepStager


def learning_rate(step: int, total_steps: int) -> float:
    """Cosine decay from the peak to 1% of it."""
    return config.LEARNING_RATE * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / total_steps)))


def main() -> None:
    if not (config.DATA_DIR / "index.csv").exists():
        sys.exit(f"No preprocessed data in {config.DATA_DIR}. Run `python preprocess.py` first.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.SEED)

    splits = split_recordings()
    print("Loading data ...")
    train = load_recordings(splits["train"])
    val = load_recordings(splits["val"])
    print(
        f"Device: {device} | recordings: {len(train)} train, {len(val)} val, {len(splits['test'])} test"
        f" | {sum(len(y) for _, y in train):,} training epochs"
    )

    sampler = BatchSampler(train, config.SEED)
    weights = class_weights(train).to(device)
    model = SleepStager().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters\n")

    total_steps = config.MAX_EPOCHS * config.STEPS_PER_EPOCH
    best_f1, epochs_since_best = 0.0, 0
    config.CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plots.RESULTS_DIR.mkdir(exist_ok=True)
    with open(plots.HISTORY_FILE, "w", newline="") as file:
        csv.writer(file).writerow(["epoch", "loss", "val_accuracy", "val_macro_f1"])
    started = time.time()

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step in range(config.STEPS_PER_EPOCH):
            global_step = (epoch - 1) * config.STEPS_PER_EPOCH + step
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(global_step, total_steps)

            x, y = sampler.sample(device)
            # Full precision on purpose: half precision (autocast) gives NaNs on GTX 16-series cards.
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), weight=weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item()

        scores = metrics(confusion_matrix(model, val, device))
        saved = ""
        if scores["macro_f1"] > best_f1:
            best_f1, epochs_since_best = scores["macro_f1"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_f1": best_f1}, config.CHECKPOINT_FILE)
            saved = " | saved"
        else:
            epochs_since_best += 1
        with open(plots.HISTORY_FILE, "a", newline="") as file:
            csv.writer(file).writerow([epoch, running_loss / config.STEPS_PER_EPOCH, scores["accuracy"], scores["macro_f1"]])
        minutes = (time.time() - started) / 60
        print(
            f"epoch {epoch:2d} | loss {running_loss / config.STEPS_PER_EPOCH:.3f}"
            f" | val accuracy {scores['accuracy']:.3f} | val macro F1 {scores['macro_f1']:.3f}"
            f" | {minutes:.1f} min{saved}",
            flush=True,
        )
        if epochs_since_best >= config.PATIENCE:
            print(f"No improvement for {config.PATIENCE} epochs, stopping early.")
            break

    print(f"\nBest validation macro F1 {best_f1:.3f}. Model saved to {config.CHECKPOINT_FILE}")
    print("\nTest set results for the saved model:")
    model.load_state_dict(torch.load(config.CHECKPOINT_FILE, map_location=device)["model"])
    print(report(confusion_matrix(model, load_recordings(splits["test"]), device)))
    print()
    plots.main()


if __name__ == "__main__":
    main()
