import matplotlib.pyplot as plt

# ScienceQA support, ScienceQA query
# Llama-3.2V-11B
# 82.85
# 84.43
# 84.13
# 83.64

# LLaVA-CoT
# 92.81
# 91.97
# 91.57
# 90.48

# Qwen2.5VL-3B
# 81.11
# 80.61
# 80.61
# 81.06

# VLM-R1
# 82.30
# 83.39
# 82.45
# 82.94

# A-OKVQA (support) ScienceQA (query) 
# Llama-3.2V-11B
# 85.32
# 84.78
# 85.32
# 84.93
# 84.48

# LLaVA-CoT
# 93.11
# 92.46
# 91.18
# 91.62

# Qwen2.5VL-3B
# 81.11
# 80.42
# 80.22
# 81.11

# VLM-R1
# 82.60
# 82.60
# 82.70 
# 82.90

# A-OKVQA (support) A-OKVQA (query)
# Llama-3.2V-11B
# 84.02
# 83.49
# 83.58
# 82.71

# LLaVA-CoT
# 85.24
# 85.58
# 84.54
# 83.23

# Qwen2.5VL-3B
# 82.01
# 80.26
# 80.96
# 78.17

# VLM-R1
# 82.71
# 82.01
# 81.48
# 81.14

# ScienceQA (support) A-OKVQA (query) 
# Llama-3.2V-11B
# 82.97
# 82.53
# 83.06
# 82.53

# LLaVA-CoT
# 84.37
# 85.24
# 83.84
# 84.37

# Qwen2.5VL-3B
# 83.23
# 82.18
# 82.10
# 82.53

# VLM-R1
# 83.93
# 82.71
# 82.71
# 82.62

import matplotlib.pyplot as plt
import numpy as np

# Settings
shots = ["1-shot", "2-shot", "4-shot", "8-shot"]
y_pos = np.arange(len(shots))

# Reordered models for paired layout
models = ["Llama-3.2V-11B", "LLaVA-CoT", "Qwen2.5VL-3B", "VLM-R1"]
colors = ['steelblue', 'lightskyblue', 'darkorange', 'gold']
offsets = [-0.3, -0.1, 0.1, 0.3]
bar_width = 0.18

# Data: IID and OOD for both query sets
scienceqa_iid = {
    "Llama-3.2V-11B": [82.85, 84.43, 84.13, 83.64],
    "LLaVA-CoT": [92.81, 91.97, 91.57, 90.48],
    "Qwen2.5VL-3B": [81.11, 80.61, 80.61, 81.06],
    "VLM-R1": [82.30, 83.39, 82.45, 82.94]
}
scienceqa_ood = {
    "Llama-3.2V-11B": [85.32, 84.78, 85.32, 84.93],
    "LLaVA-CoT": [93.11, 92.46, 91.18, 91.62],
    "Qwen2.5VL-3B": [81.11, 80.42, 80.22, 81.11],
    "VLM-R1": [82.60, 82.60, 82.70, 82.90]
}

aokvqa_iid = {
    "Llama-3.2V-11B": [84.02, 83.49, 83.58, 82.71],
    "LLaVA-CoT": [85.24, 85.58, 84.54, 83.23],
    "Qwen2.5VL-3B": [82.01, 80.26, 80.96, 78.17],
    "VLM-R1": [82.71, 82.01, 81.48, 81.14]
}
aokvqa_ood = {
    "Llama-3.2V-11B": [82.97, 82.53, 83.06, 82.53],
    "LLaVA-CoT": [84.37, 85.24, 83.84, 84.37],
    "Qwen2.5VL-3B": [83.23, 82.18, 82.10, 82.53],
    "VLM-R1": [80.61, 79.65, 79.21, 79.13]
}

# Compute differences
def compute_diff(ood, iid):
    return [np.array(ood[m]) - np.array(iid[m]) for m in models]

diff_scienceqa = compute_diff(scienceqa_ood, scienceqa_iid)
diff_aokvqa = compute_diff(aokvqa_ood, aokvqa_iid)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

# --- ScienceQA Query ---
ax = axes[0]
for i, (vals, label, color, offset) in enumerate(zip(diff_scienceqa, models, colors, offsets)):
    ax.barh(y_pos + offset, vals, height=bar_width, color=color, label=label)

ax.set_title("ScienceQA Query", fontsize=18)
ax.set_yticks(y_pos)
ax.set_yticklabels(shots, fontsize=14)
ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel("OOD - IID Accuracy (%)", fontsize=14)
ax.legend(fontsize=12)

# --- A-OKVQA Query ---
ax = axes[1]
for i, (vals, label, color, offset) in enumerate(zip(diff_aokvqa, models, colors, offsets)):
    ax.barh(y_pos + offset, vals, height=bar_width, color=color, label=label)

ax.set_title("A-OKVQA Query", fontsize=18)
ax.set_yticks(y_pos)
ax.set_yticklabels(shots, fontsize=14)
ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel("OOD - ID Accuracy (%)", fontsize=14)
ax.legend(fontsize=12)

# fig.suptitle("Cross-Dataset Generalization Gap (OOD - IID)", fontsize=18)
plt.tight_layout()
plt.show()
plt.savefig("draw/ood_id.png", dpi=300)

# import matplotlib.pyplot as plt
# import numpy as np

# # Categories
# datasets = [
#     "ImageNet-R", "ObjectNet", "ImageNet Sketch",
#     "ImageNet-A", "ImageNet Vid", "Youtube-BB",
#     "ImageNetV2", "ImageNet"
# ]

# # Values for two different settings (e.g., Model A and Model B)
# model_a = [-4.7, -3.8, -2.8, -1.9, -0.5, 0.6, 5.8, 9.2]
# model_b = [-3.9, -3.2, -2.5, -1.5, -0.2, 0.9, 6.2, 10.1]

# # Bar settings
# y = np.arange(len(datasets))
# bar_height = 0.35

# # Plot
# fig, ax = plt.subplots(figsize=(9, 5))
# bars_a = ax.barh(y - bar_height/2, model_a, height=bar_height, color='orange', edgecolor='black', label='Model A')
# bars_b = ax.barh(y + bar_height/2, model_b, height=bar_height, color='red', edgecolor='black', label='Model B')

# # Text annotations
# for bars, data in zip([bars_a, bars_b], [model_a, model_b]):
#     for bar, val in zip(bars, data):
#         offset = 0.3 if val > 0 else -0.3
#         ha = 'left' if val > 0 else 'right'
#         ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:+.1f}", va='center', ha=ha, fontsize=9)

# # Formatting
# ax.set_yticks(y)
# ax.set_yticklabels(datasets)
# ax.axvline(x=0, color='black')
# ax.set_xlim(-10, 30)
# ax.set_xlabel("Change from zero-shot ImageNet classifier accuracy (%)")
# ax.set_title("Adapt to ImageNet", fontsize=14, color='red')

# # Remove outer box
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)

# # Add legend
# ax.legend(loc='lower right', frameon=False)

# plt.tight_layout()
# plt.savefig("draw/ood_id.png", dpi=300)
