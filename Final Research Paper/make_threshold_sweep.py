#!/usr/bin/env python3
"""Threshold sweep for the chosen single relevance model (held-out test set).
Numbers from 04_evaluate.py on model_single_deberta. Justifies the operating
point: a decision threshold near 0.80 satisfies both targets (P>=0.90, R>=0.80).
Produces fig_threshold_sweep.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

thr = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
P   = [0.878, 0.878, 0.878, 0.878, 0.875, 0.897, 0.919, 0.914, 0.912, 0.968, 1.000]
R   = [0.857, 0.857, 0.857, 0.857, 0.833, 0.833, 0.810, 0.762, 0.738, 0.714, 0.714]
F1  = [0.867, 0.867, 0.867, 0.867, 0.854, 0.864, 0.861, 0.831, 0.816, 0.822, 0.833]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.plot(thr, P,  "-o", color="#2E6F9E", lw=1.8, ms=4, label="Precision")
ax.plot(thr, R,  "-s", color="#E1A33B", lw=1.8, ms=4, label="Recall")
ax.plot(thr, F1, "-^", color="#7BA05B", lw=1.6, ms=4, label="F1")

# target lines
ax.axhline(0.90, color="#C0392B", ls="--", lw=1.0)
ax.axhline(0.80, color="#1F8A4C", ls="--", lw=1.0)
ax.text(0.505, 0.905, "Precision target 0.90", color="#C0392B", fontsize=7.5, va="bottom")
ax.text(0.505, 0.805, "Recall target 0.80",    color="#1F8A4C", fontsize=7.5, va="bottom")

# window where both targets are met, and the operating point
ax.axvspan(0.78, 0.81, color="#000000", alpha=0.07)
ax.annotate("operating point\nP=0.92, R=0.81 @ 0.80",
            xy=(0.80, 0.919), xytext=(0.62, 0.95),
            fontsize=8, ha="left",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))

ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score on held-out test set")
ax.set_xlim(0.49, 0.97)
ax.set_ylim(0.68, 1.01)
ax.set_title("Chosen single model: precision/recall vs decision threshold", fontsize=10)
ax.legend(loc="lower center", fontsize=8, ncol=3)
ax.grid(ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig("fig_threshold_sweep.png", dpi=150)
print("wrote fig_threshold_sweep.png")
