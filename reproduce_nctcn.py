#!/usr/bin/env python3
"""One-command reproduction of NC-TCN paper results.

Paper:
  J. H. Choi, "Noise-Conditional Temporal Convolutional Networks for
  Robust On-Device Keyword Spotting," Proc. ICASSP 2027 (submitted);
  MLSP 2026 workshop version.

What this script does (for reviewers):
  1. Environment check (torch, numpy, scipy).
  2. Locates Google Speech Commands v0.02 (auto-downloads on Colab).
  3. Loads the shipped checkpoint `checkpoints/nc_tcn_20k_best.pt`
     (21,689 params).
  4. Evaluates clean + 5 noise types x 3 SNRs (or a subset if --quick).
  5. Optionally runs with spectral-subtraction preprocessing (--ss).
  6. Prints a measured-vs-paper table. Exits 0 on match within --tol.

Canonical training notebook (Colab):
    https://colab.research.google.com/github/DrJinHoChoi/NC-TCN/blob/main/notebooks/train_nc_tcn.ipynb

Usage:
    python reproduce_nctcn.py              # quick: clean + factory @ 0dB
    python reproduce_nctcn.py --full       # all noise x SNR grid
    python reproduce_nctcn.py --ss         # with spectral subtraction
    python reproduce_nctcn.py --data-dir /path/to/SpeechCommands

Exit codes: 0 pass, 1 numeric mismatch, 2 deps missing,
            3 dataset missing, 4 checkpoint missing.

License: see LICENSE / COMMERCIAL_LICENSE.md.
Author: Jin Ho Choi (SmartEAR) -- jinhochoi@smartear.co.kr
"""

from __future__ import annotations
import argparse
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "eval"))


# --------------------------------------------------------------------- #
# Paper reference numbers (placeholder -- update with ICASSP 2027 tex)  #
# --------------------------------------------------------------------- #
# Key: (noise, snr_db) -> accuracy %
PAPER_TABLE = {
    # Clean
    ("clean", None): 95.0,
    # Factory
    ("factory", 5):  84.0,
    ("factory", 0):  72.0,
    ("factory", -5): 55.0,
    # Babble
    ("babble", 5):   83.0,
    ("babble", 0):   70.0,
    ("babble", -5):  52.0,
    # White
    ("white", 5):    85.0,
    ("white", 0):    73.0,
    ("white", -5):   56.0,
    # Pink
    ("pink", 5):     84.0,
    ("pink", 0):     72.0,
    ("pink", -5):    55.0,
    # Street
    ("street", 5):   83.0,
    ("street", 0):   71.0,
    ("street", -5):  53.0,
}


def check_env() -> None:
    print("=" * 72)
    print("NC-TCN -- ICASSP 2027 / MLSP 2026 reproduction")
    print("=" * 72)
    print(f"Python   : {platform.python_version()}")
    print(f"Platform : {platform.platform()}")
    missing = []
    for pkg in ("numpy", "scipy", "torch"):
        try:
            m = __import__(pkg)
            print(f"{pkg:<9}: {getattr(m, '__version__', '?')}")
        except ImportError:
            missing.append(pkg)
            print(f"{pkg:<9}: MISSING")
    if missing:
        print(f"\nMissing: {missing}. Install: pip install numpy scipy torch")
        sys.exit(2)
    import torch
    print(f"CUDA     : {torch.cuda.is_available()} "
          f"({torch.cuda.device_count()} GPU)")
    print("=" * 72)


def locate_dataset(data_dir: Path | None) -> Path:
    candidates = []
    if data_dir is not None:
        candidates.append(data_dir)
    candidates += [
        REPO / "data" / "SpeechCommands" / "speech_commands_v0.02",
        REPO.parent / "SpeechCommands" / "speech_commands_v0.02",
        Path.home() / "SpeechCommands" / "speech_commands_v0.02",
    ]
    for c in candidates:
        if c.exists() and (c / "testing_list.txt").exists():
            print(f"\n[OK] GSC v0.02 found: {c}")
            return c
    print("\n[ERROR] Google Speech Commands v0.02 not found. Searched:")
    for c in candidates:
        print(f"  - {c}")
    print("\nDownload:")
    print("  wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz")
    print("  mkdir -p data/SpeechCommands/speech_commands_v0.02")
    print("  tar -xzf speech_commands_v0.02.tar.gz -C data/SpeechCommands/speech_commands_v0.02/")
    sys.exit(3)


def load_model(ckpt_path: Path):
    import torch
    from nanomamba import create_nc_tcn_20k
    if not ckpt_path.exists():
        print(f"\n[ERROR] Checkpoint not found: {ckpt_path}")
        print("Train via: notebooks/train_nc_tcn.ipynb (Colab)")
        sys.exit(4)
    model = create_nc_tcn_20k()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[OK] NC-TCN-20K loaded: {n:,} params ({n/1024:.1f} KB INT8)")
    return model


def run_eval(model, samples, noise_type, snr_db, use_ss, batch=128):
    """Delegates to eval_nctcn.evaluate for consistency with paper."""
    from eval_nctcn import evaluate
    return evaluate(model, samples, batch_size=batch,
                    noise_type=noise_type, snr_db=snr_db, use_ss=use_ss)


def report(measured: dict, tol: float) -> int:
    print("\n" + "=" * 72)
    print("NC-TCN paper reproduction")
    print("=" * 72)
    print(f"{'Noise':<10} {'SNR':>6} {'measured':>10} "
          f"{'paper':>10} {'|delta|':>10}")
    print("-" * 72)
    max_dev = 0.0
    for (noise, snr), m in measured.items():
        p = PAPER_TABLE.get((noise, snr))
        if p is None:
            print(f"{noise:<10} {str(snr):>6} {m:>10.1f} "
                  f"{'n/a':>10} {'--':>10}")
            continue
        d = abs(m - p)
        max_dev = max(max_dev, d)
        flag = "" if d <= tol else " !"
        print(f"{noise:<10} {str(snr):>6} {m:>10.1f} {p:>10.1f} "
              f"{d:>+10.2f}{flag}")
    print("=" * 72)
    print(f"Max |delta| vs paper = {max_dev:.2f}  (tol = {tol})")
    if max_dev <= tol:
        print("[PASS] reproduction within tolerance.")
        return 0
    print("[FAIL] reproduction outside tolerance.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="NC-TCN reviewer reproduction")
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "checkpoints" / "nc_tcn_20k_best.pt")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--full", action="store_true",
                    help="Full noise x SNR grid (slow).")
    ap.add_argument("--ss", action="store_true",
                    help="Apply spectral subtraction preprocessing.")
    ap.add_argument("--quick", action="store_true",
                    help="Clean + factory @ 0dB only (smoke test).")
    ap.add_argument("--tol", type=float, default=3.0,
                    help="Accuracy tolerance (percentage points).")
    args = ap.parse_args()

    check_env()
    t0 = time.time()

    # Redirect DATA_ROOT used by eval_nctcn to user's path
    data_root = locate_dataset(args.data_dir)
    import eval_nctcn
    eval_nctcn.DATA_ROOT = data_root

    # Reload samples from chosen DATA_ROOT
    samples = eval_nctcn.load_test_set()
    print(f"Test samples: {len(samples)}")

    model = load_model(args.ckpt)

    # Determine grid
    if args.quick or not args.full:
        grid = [("clean", None), ("factory", 0)]
    else:
        grid = [("clean", None)]
        for n in ("factory", "babble", "white", "pink", "street"):
            for s in (5, 0, -5):
                grid.append((n, s))

    measured = {}
    for noise, snr in grid:
        nt = None if noise == "clean" else noise
        sd = None if snr is None else snr
        acc, _ = run_eval(model, samples, nt, sd, args.ss)
        measured[(noise, snr)] = acc * 100
        tag_ss = " +SS" if args.ss else ""
        print(f"  [{noise:<8} SNR={str(snr):>4}{tag_ss}] acc = "
              f"{acc*100:.1f}%")

    code = report(measured, tol=args.tol)
    print(f"\nTotal elapsed: {time.time()-t0:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
