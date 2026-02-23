# Multi-Modal Fusion for Automatic Modulation Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)]()

> A rigorous ablation study quantifying the contribution of each input modality — raw IQ signals, constellation diagrams, and spectrograms — to deep learning-based automatic modulation classification (AMC).

---

## Overview

This repository contains the full implementation of a multi-modal deep learning framework for AMC, including a complete ablation study comparing **7 branch configurations** across **5 independent seeds** and **100 training epochs**.

### Modulation Schemes
`BPSK` · `QPSK` · `8PSK` · `16QAM` · `64QAM`

### SNR Range
−20 dB to +20 dB (step 2 dB, 21 levels)

---

## Ablation Configurations

| # | Configuration | IQ Branch | Constellation Branch | Spectrogram Branch |
|---|---------------|:---------:|:--------------------:|:------------------:|
| 1 | IQ only        | ✓ | | |
| 2 | Const only     | | ✓ | |
| 3 | Spec only      | | | ✓ |
| 4 | IQ + Const     | ✓ | ✓ | |
| 5 | IQ + Spec      | ✓ | | ✓ |
| 6 | Const + Spec   | | ✓ | ✓ |
| 7 | Full Fusion    | ✓ | ✓ | ✓ |

---

## Repository Structure

```
amc-multimodal-ablation/
├── ablation_kaggle.ipynb      ← Main notebook (Kaggle / GPU ready)
├── requirements.txt           ← Python dependencies
├── LICENSE                    ← MIT License
├── .gitignore
│
├── figures/                   ← Auto-generated plots (populated after training)
│   ├── ablation_bar_chart.png
│   ├── per_snr_accuracy.png
│   ├── accuracy_heatmap.png
│   └── learning_curves_*.png
│
├── results/                   ← Exported numerical results
│   ├── results_summary.csv
│   ├── per_snr_accuracy.csv
│   └── results_full.json
│
└── saved_models/              ← Model checkpoints (not tracked by git)
    └── {config}_seed{n}.pt
```

---

## Quickstart

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/amc-multimodal-ablation.git
cd amc-multimodal-ablation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on Kaggle (recommended)
1. Go to [kaggle.com](https://kaggle.com) → **Create → New Notebook**
2. **File → Import Notebook** → upload `ablation_kaggle.ipynb`
3. Set **Accelerator → GPU T4** in the right panel
4. **Save Version → Save & Run All (Commit)** to run in background

### 4. Run locally
```bash
jupyter notebook ablation_kaggle.ipynb
```
Run cells 1–6 to set up, then run each Section 7 config cell.
The notebook **auto-resumes** from saved checkpoints if interrupted.

---

## Model Architecture

### Three Input Branches

**IQ Encoder** — 1D CNN on raw signal
```
Input (2, 128) → Conv1d×3 [64→128→256] + BN + ReLU + MaxPool → GAP → FC(256→128)
```

**Constellation Encoder** — 2D CNN on IQ scatter histogram
```
Input (3, 64, 64) → Conv2d×3 [32→64→128] + BN + ReLU + MaxPool → GAP → FC(128→128)
```

**Spectrogram Encoder** — 2D CNN on magnitude spectrogram
```
Input (3, 64, 64) → Conv2d×3 [32→64→128] + BN + ReLU + MaxPool → GAP → FC(128→128)
```

### Fusion Head
```
Cat(active branches) → Linear(n×128, 256) → ReLU → Dropout(0.5) → Linear(256, 5)
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 256 |
| Optimiser | Adam (lr = 1e-3) |
| LR scheduler | ReduceLROnPlateau (patience=7, factor=0.5) |
| Seeds | 5 (0–4) |
| Train / Test split | 70 / 30 (stratified by class) |
| Samples per mod per SNR | 500 |
| Total dataset size | 52,500 samples |

---

## Results

*Results will be populated here after training completes.*

| Configuration | Mean Acc | Std |
|---|---|---|
| IQ only | — | — |
| Const only | — | — |
| Spec only | — | — |
| IQ + Const | — | — |
| IQ + Spec | — | — |
| Const + Spec | — | — |
| Full Fusion | — | — |

Full per-SNR breakdown available in [`results/per_snr_accuracy.csv`](results/per_snr_accuracy.csv).

---

## Reproducing Results

Load any saved checkpoint:

```python
# Inside the notebook after running Section 6
model, ckpt = load_checkpoint("Full Fusion", seed=0)
print(f"Best accuracy: {ckpt['best_acc']*100:.2f}%")

# Replay learning curves
plot_learning_curves(ckpt["config"], [ckpt["train_losses"]], [ckpt["val_accs"]], save=False)
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{amc_multimodal_ablation_2025,
  title   = {Multi-Modal Fusion for Automatic Modulation Classification: An Ablation Study},
  author  = {YOUR NAME},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/amc-multimodal-ablation}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
