"""BTPC: Barlow Twins self-supervised P-wave polarity classification.

A two-stage, label-free pipeline for seismic P-wave polarity:

* Stage 1 trains a Barlow Twins encoder on two augmented views of each
  P-wave snippet and periodically filters training samples with an
  unsupervised reliability score.
* Stage 2 clusters encoder features (spectral clustering by default) into
  two pseudo-classes and trains a classification head on them.
* Validation and prediction map the A/B pseudo-classes to physical up/down
  polarities with a small labelled anchor set and reject unstable
  predictions.

Command line entry points: ``btpc-train``, ``btpc-valid``, ``btpc-predict``.
"""

import os

# All figures are written to PNG files, never shown interactively, so the
# pipeline must run on headless machines (servers, CI, WSL without an X
# server). Respect an explicit MPLBACKEND if the user set one.
os.environ.setdefault("MPLBACKEND", "Agg")

__version__ = "0.1.0"
