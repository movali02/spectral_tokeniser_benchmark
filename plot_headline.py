"""Headline figure for the README: SpectraLLM's format (A) against the full string (B_modes_on).

Values are mean ECFP4 Tanimoto on the 1,000 test molecules (sampling decoding, invalid = 0),
from experiment_1_phase6/summary_test.csv. Writes assets/full_string_vs_spectrallm.png.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = ["IR", "Raman", "IR + Raman"]
A          = [0.194, 0.193, 0.235]   # SpectraLLM's format
B_MODES_ON = [0.293, 0.264, 0.349]   # our format + mode labels
FLOOR = 0.081                        # a random training molecule

INK, MUTED, GRID = "#1F2227", "#6B7178", "#E3E5E8"
C_A, C_B = "#4F7CC0", "#C4420E"

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(CONDITIONS)); w = 0.36
ba = ax.bar(x - w/2 - 0.01, A, w, color=C_A, label="SpectraLLM's format (A)")
bb = ax.bar(x + w/2 + 0.01, B_MODES_ON, w, color=C_B, label="Our format + mode labels (B_modes_on)")
for bars in (ba, bb):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.006, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=10, color=INK)
ax.axhline(FLOOR, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=3,
           label="Random training molecule (0.081)")

ax.set_xticks(x); ax.set_xticklabels(CONDITIONS, color=INK)
ax.set_ylabel("Mean ECFP4 Tanimoto", color=MUTED)
ax.set_ylim(0, 0.40); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
ax.tick_params(axis="y", colors=MUTED, length=0); ax.tick_params(axis="x", length=0)
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.2), ncol=2, fontsize=9.5,
          handlelength=1.2, columnspacing=1.6)
fig.tight_layout()
Path("assets").mkdir(exist_ok=True)
fig.savefig("assets/full_string_vs_spectrallm.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote assets/full_string_vs_spectrallm.png")
