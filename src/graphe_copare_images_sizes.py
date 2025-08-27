import matplotlib.pyplot as plt
import numpy as np

# Données
labels = ["1square", "2square", "4square", "rectangle", "normal"]

# Moyennes (ici 2 dernières = fausses valeurs)
mae = [9.417, 10.182, 8.939, 10.152, 8.062]

# Écarts-types (2 dernières = fausses valeurs aussi)
std = [1.965, 1.78, 2.099,1.197, 0.466]

x = np.arange(len(labels))
width = 0.6

fig, ax = plt.subplots(figsize=(7,4))
bars = ax.bar(x, mae, width, yerr=std, capsize=5, 
              color=["#4C72B0","#55A868","#C44E52","#8172B3","#CCB974"])

# Ajout des valeurs au-dessus de chaque barre
for bar, value in zip(bars, mae):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f"{value:.2f}", ha='center', va='bottom', fontsize=10)

# Mise en forme
ax.set_ylabel("Error (MAE)")
ax.set_xlabel("Crop size")
ax.set_title("Performance according to size")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(mae) + 3)

plt.tight_layout()
plt.show()
