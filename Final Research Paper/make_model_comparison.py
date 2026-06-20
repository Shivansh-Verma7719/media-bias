#!/usr/bin/env python3
"""Model comparison chart for the relevance-classifier methodology section.
All numbers measured on the locked 187-article test set (test.csv), each model
at its F1-optimal threshold (04_evaluate.py threshold sweep). The 3-model
ensemble is shown for reference only; it was abandoned in favour of the single
consolidated model.
Produces fig_model_comparison.png in this folder.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = [
    "Labeled\nonly",
    "+ Synthetic\n(broad)",
    "+ Synthetic\n(targeted, focal)",
    "Single model\n(chosen)",
    "Ensemble\n(reference)",
]
precision = [0.652, 0.780, 0.860, 0.878, 0.944]
recall    = [0.714, 0.762, 0.881, 0.857, 0.810]
f1        = [0.682, 0.771, 0.871, 0.867, 0.872]

x = np.arange(len(models))
w = 0.26

fig, ax = plt.subplots(figsize=(8.2, 4.5))
b1 = ax.bar(x - w, precision, w, label="Precision", color="#2E6F9E")
b2 = ax.bar(x,     recall,    w, label="Recall",    color="#E1A33B")
b3 = ax.bar(x + w, f1,        w, label="F1",        color="#7BA05B")

# target lines
ax.axhline(0.90, color="#C0392B", ls="--", lw=1.0)
ax.axhline(0.80, color="#1F8A4C", ls="--", lw=1.0)
ax.text(len(models)-0.45, 0.905, "Precision target 0.90", color="#C0392B", fontsize=7, va="bottom", ha="right")
ax.text(len(models)-0.45, 0.805, "Recall target 0.80",    color="#1F8A4C", fontsize=7, va="bottom", ha="right")

# highlight chosen model (index 3)
ax.axvspan(2.55, 3.45, color="#000000", alpha=0.06)

for bars in (b1, b2, b3):
    for r in bars:
        ax.annotate(f"{r.get_height():.2f}", (r.get_x()+r.get_width()/2, r.get_height()),
                    ha="center", va="bottom", fontsize=6.2, xytext=(0, 1), textcoords="offset points")

ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
ax.set_ylabel("Score on held-out test set (F1-optimal threshold)")
ax.set_ylim(0.5, 1.0)
ax.set_title("Relevance classifier: held-out performance across training strategies", fontsize=10)
ax.legend(loc="lower left", fontsize=8, ncol=3)
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig("fig_model_comparison.png", dpi=150)
print("wrote fig_model_comparison.png")
