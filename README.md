# NC-TCN

**Noise-Conditional Temporal Convolutional Networks for Robust On-Device Keyword Spotting**

Target venues: ICASSP 2027 · MLSP 2026

---

## Overview

NC-TCN is an ultra-lightweight (~20K params) keyword-spotting model that conditions temporal convolutional blocks on estimated noise statistics for robustness in low-SNR conditions.

---

## Repository Layout

```
NC-TCN/
├── notebooks/
│   └── train_nc_tcn.ipynb          # Colab-style training notebook
├── eval/
│   └── eval_nctcn.py               # CPU noise-robustness evaluation
├── checkpoints/
│   └── nc_tcn_20k_best.pt          # Trained 20K-param model weights
├── src/
│   ├── nanomamba.py                # Model factory (contains create_nc_tcn_20k)
│   └── train_colab.py              # Noise mixing + spectral subtraction utilities
├── icassp2027_nctcn.pdf            # ICASSP 2027 submission
├── mlsp2026_nctcn.pdf              # MLSP 2026 workshop submission
└── README.md
```

> **Note**: The model definition currently resides in `src/nanomamba.py` (shared with the NC-SSM / NanoMamba family). A standalone refactor into `src/nc_tcn/` is planned before the camera-ready version.

---

## Quick Start

```bash
# Assumes Google Speech Commands v0.02 at data/SpeechCommands/
python eval/eval_nctcn.py
```

---

## Citation

```bibtex
@inproceedings{choi2027nctcn,
  title={Noise-Conditional Temporal Convolutional Networks for Robust On-Device Keyword Spotting},
  author={Choi, Jin Ho},
  booktitle={ICASSP 2027},
  year={2027}
}
```

---

## Related Work

- **NC-SSM / NanoMamba** (IEEE TASLP 2026) — [NC-SSM-TASLP2026](https://github.com/DrJinHoChoi/NC-SSM-TASLP2026)
- **NC-SLU** (EMNLP 2026) — [EMNLP2026-NC-SLU](https://github.com/DrJinHoChoi/EMNLP2026-NC-SLU)
- **NC-OPAL** (Few-Shot KWS) — [NC-KWS-FineTuning](https://github.com/DrJinHoChoi/NC-KWS-FineTuning)
- **NC-Conv-SSM** (Vision) — [NC-SSM-Vision-ICCV2027](https://github.com/DrJinHoChoi/NC-SSM-Vision-ICCV2027)

---

**Author**: Jin Ho Choi, Ph.D. — [DrJinHoChoi.github.io](https://drjinhochoi.github.io)
