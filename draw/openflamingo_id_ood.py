import matplotlib.pyplot as plt
import numpy as np

# New shot settings and positions
shots = ["4-shot", "8-shot", "16-shot", "32-shot"]
y_pos = np.arange(len(shots))
bar_width = 0.18

# Model data
models = ["OKVQA Query", "TextVQA Query"]
colors = ['steelblue', 'darkorange']

# OKVQA data
okvqa_id = np.array([30.04, 30.67, 31.15, 30.76])
okvqa_ood = np.array([25.60, 25.94, 25.52, 24.76])
okvqa_diff = okvqa_ood - okvqa_id

# TextVQA data
textvqa_id = np.array([27.33, 28.02, 27.92, 28.51])
textvqa_ood = np.array([26.49, 26.49, 26.64, 26.33])
textvqa_diff = textvqa_ood - textvqa_id

# Plotting
fig, ax = plt.subplots(figsize=(7, 4))

offsets = [-0.1, 0.1]
for i, (diff, label, color, offset) in enumerate(zip(
    [okvqa_diff, textvqa_diff], models, colors, offsets
)):
    ax.barh(y_pos + offset, diff, height=bar_width, color=color, label=label)

# Formatting
# ax.set_title("OpenFlamingo-3B: OOD - IID Accuracy", fontsize=18)
ax.set_yticks(y_pos)
ax.set_yticklabels(shots, fontsize=14)
ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel("OOD - ID Accuracy (%)", fontsize=14)
ax.legend(fontsize=12, loc='lower left')

plt.tight_layout()
plt.show()
plt.savefig("draw/openflamingo_ood_id.png", dpi=300)
