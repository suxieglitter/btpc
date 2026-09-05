"""Generate a tiny synthetic SCSN-style dataset + anchor bank for smoke tests.

The output mimics the real data layout (600-sample traces, P arrival at
index 300, keys ``X``/``Y``/``snr``) so the full btpc pipeline can be exercised
end-to-end without downloading anything:

    python scripts/make_toy_data.py --output-dir runs/toy
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from btpc.anchor_utils import build_anchor_bank
except ImportError:  # run from a fresh checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from btpc.anchor_utils import build_anchor_bank


def make_waveforms(n_samples, rng, length=600, p_index=300):
    """Damped-sine P bursts of random polarity on unit-variance noise.

    The stored ``snr`` equals the burst amplitude, i.e. roughly the
    max|post-P| / max|pre-P| ratio used by the real datasets.
    """
    snr = 10.0 ** rng.uniform(0.0, 3.0, size=n_samples)
    labels = rng.choice([0, 1, 2], size=n_samples, p=[0.475, 0.475, 0.05])
    sign = np.where(labels == 1, -1.0, 1.0)
    sign[labels == 2] = rng.choice([-1.0, 1.0], size=int(np.sum(labels == 2)))

    noise = rng.standard_normal((n_samples, length))
    burst_len = 40
    t = np.arange(burst_len)
    envelope = np.exp(-t / 12.0)
    carrier = np.sin(2.0 * np.pi * 0.15 * t)
    burst = sign[:, None] * snr[:, None] * envelope[None, :] * carrier[None, :]

    waveforms = noise.astype(np.float32)
    waveforms[:, p_index : p_index + burst_len] += burst.astype(np.float32)
    return waveforms, labels.astype(np.int64), snr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-pool-per-label", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    waveforms, labels, snr = make_waveforms(args.num_samples, rng)
    data_path = out_dir / "toy_scsn.hdf5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("X", data=waveforms)
        handle.create_dataset("Y", data=labels)
        handle.create_dataset("snr", data=snr)
    print(f"Saved toy dataset to: {data_path}")
    print(
        f"  samples={len(labels)} | up={int(np.sum(labels == 0))} "
        f"down={int(np.sum(labels == 1))} uncertain={int(np.sum(labels == 2))}"
    )

    bank_path = out_dir / "toy_anchor_bank.npz"
    build_anchor_bank(
        bank_path=str(bank_path),
        data_path=str(data_path),
        snr_range=(50.0, 1000.0),
        pool_per_label=args.anchor_pool_per_label,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
