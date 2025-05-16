import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Shot settings
shots = ["4-shot", "8-shot", "16-shot", "32-shot"]
x = np.arange(len(shots))
bar_width = 0.2

# Data: (ID, OOD)
data = {
    "Random": (np.array([30.04, 30.67, 31.15, 30.76]), np.array([25.60, 25.94, 25.52, 24.76])),
    "RICES":  (np.array([30.97, 32.39, 33.46, 34.44]), np.array([22.94, 23.94, 23.12, 22.97])),
    "JICES":  (np.array([34.69, 35.79, 35.88, 36.44]), np.array([21.75, 20.01, 20.48, 20.19]))
}

# Updated color scheme: base = OOD (light), delta = ID - OOD (dark)
colors_base = {
    "Random": "lightskyblue",
    "RICES": "moccasin",
    "JICES": "plum"
}
colors_delta = {
    "Random": "steelblue",
    "RICES": "darkorange",
    "JICES": "orchid"
}

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
offsets = [-bar_width, 0, bar_width]

for i, (method, (id_vals, ood_vals)) in enumerate(data.items()):
    for j in range(len(shots)):
        id_score = id_vals[j]
        ood_score = ood_vals[j]
        delta = max(id_score - ood_score, 0)

        # Base bar (always OOD)
        ax.bar(
            x[j] + offsets[i],
            ood_score,
            width=bar_width,
            color=colors_base[method]
        )

        # Delta overlay (only if ID - OOD)
        if delta > 0:
            ax.bar(
                x[j] + offsets[i],
                delta,
                bottom=ood_score,
                width=bar_width,
                color=colors_delta[method]
            )

# Axis labels
ax.set_xticks(x)
ax.set_xticklabels(shots, fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=14)
# ax.set_title("ID vs. OOD (Delta Over OOD Base) — TextVQA → OK-VQA", fontsize=16)

# Legend
legend_elements = [
    Patch(facecolor='lightskyblue', label='Random (OOD)'),
    Patch(facecolor='steelblue', label='Random (ID - OOD)'),
    Patch(facecolor='moccasin', label='Uni-modal (OOD)'),
    Patch(facecolor='darkorange', label='Uni-modal (ID - OOD)'),
    Patch(facecolor='plum', label='Multi-modal (OOD)'),
    Patch(facecolor='orchid', label='Multi-modal (ID - OOD)')
]
ax.legend(
    handles=legend_elements,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    fontsize=8,
    frameon=False
)
# ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
plt.savefig("draw/textvqa_support_okvqa_delta_oodbase.png", dpi=300)


# ax.legend(
#     loc='upper center',
#     bbox_to_anchor=(0.5, 1.12),
#     ncol=3,
#     fontsize=10,
#     frameon=False
# )