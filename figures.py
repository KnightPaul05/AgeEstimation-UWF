import matplotlib.pyplot as plt
#Age distributions
DR=[55,55,46,46,74,74,41,41,67,67,80,80,55,55,59,59,71,71,66,66,56,56,34,34,75,75,62,62,60,61,61,61,61,68,68,63,58,58,55,55,38,38,45,45,59,59,39,39,64,64,49,69,69,66,66,63,63,60,50,50,57,57,61,61,47,47,45,45,42,42,50,50,42,42,54,52,52,60,60,75,75,76,76,38,38,78,78,56,56,68,68,69,69,45,45,60,60,53,53,34]
val_DR = 24.136
AMD = [49,49,82,82,68,68,68,68,68,65,65,76,76,67,67,86,74,74,78,62,62,56,56,66,66,72,72,86,87,87,60,60,71,71,71,71,73,75,75,66,66,66,66,71,71,82,82,69,69,80,80,82,81,81,81,77,77,77,72,72,78,78,73,73,78,60,60,60,60,66,66,87,87,54,54,69,58,92,78,58,78,55,55,67,67,77,77,63,63,75,75,66,66,78,78,73,73,62,75,75]
val_AMD =36.756
RVO=[53,69,75,53,73,59,68,72,80,81,76,74,60,65,85,73,63,43,63,66,75,57,61,63,52,67,65,67,58,79,59,74,88,69,67,81,51,67,67,66,54,63,61,66,52,56,48,55,89,65,59,69,63,62,60,60,48,71,57,65,57,46,55,56,29,73,57,88,66,62,73,84,55,78,64,54,68,61,66,49,49,65,67,50,62,81,63,56,56,58,69,75,67,67,53,66,70,62,51,66]
val_RVO =29.44
PM=[34,34,47,69,30,69,69,36,37,48,66,62,68,38,39,55,52,53,65,55,52,39,54,51,46,35,44,35,69,48,52,65,63,43,44,84,47,80,59,18,71,2,64,76,65,39,60,21,34,59,43,55,42,53,73,68,65,48,65,57,49,58,46,61,47,75,58,67,54,49,65,54,46,54,54,55,52,78,50,60,49,60,78,33,29,58,57,62,41,62,76,56,43,43,53,67,47,72,60,67]
val_PM =21.826
Uveitis=[46,58,49,45,18,38,70,33,35,33,47,34,50,39,73,73,44,36,75,50,21,52,41,31,66,70,45,65,28,56,47,62,24,63,10,48,33,49,30,52,39,38,52,64,42,64,36,32,24,34,14,57,30,64,32,20,33,31,35,26,28,59,64,47,15,48,28,42,19,22,47,57,27,43,39,29,69,71,26,61,42,69,67,60,32,80,45,26,64,50,37,45,43,34,26,67,44,42,74,57]
val_Uveitis =14.846
RD=[76,35,47,54,49,66,82,50,60,69,30,41,27,39,72,47,34,34,70,30,54,64,45,55,44,71,45,16,52,23,38,66,42,72,40,15,29,50,57,25,73,51,58,3,64,54,56,59,68,30,65,60,37,64,38,58,41,68,56,75,22,49,24,68,60,42,57,32,58,58,68,51,19,36,57,51,17,25,18,29,67,64,26,29,56,59,56,21,27,31,29,44,36,47,74,33,46,68,8,60]
val_RD =17.843
Healthy=[63,43,33,22,41,59,42,30,22,19,39,33,17,39,41,32,53,38,31,37,23,30,31,32,24,14,33,52,48,40,52,57,31,31,13,8,23,35,33,26,23,54,45,59,30,29,43,24,27,47,52,60,33,62,26,13,19,35,44,44,27,59,36,8,67,60,42,46,44,9,35,36,23,40,28,42,45,32,58,19,65,21,36,38,50,39,34,29,17,67,63,22,39,36,8,35,42,17,24,63]
val_Healthy = 10.061

bins = list(range(0, 101, 10))

#Display age distribution
def figure_age_distribution(plot):
    
    plt.hist(plot, bins=bins, alpha=0.5, label='Diabetic Retinopathy', color='blue', edgecolor='black')
    plt.hist(Healthy, bins=bins, alpha=0.5, label='Healthy', color='white', edgecolor='red')

    plt.xlabel("Age range (years)")
    plt.ylabel("Number of individuals")
    plt.title("Age distribution (10-year intervals)")


    plt.show()

#figure_age_distribution(DR)
#figure_age_distribution(AMD)
#figure_age_distribution(RVO)
#figure_age_distribution(PM)
#figure_age_distribution(Uveitis)
#figure_age_distribution(RD)
#figure_age_distribution(Healthy)

#Jensen-Shannon divergence
from scipy.spatial.distance import jensenshannon
import numpy as np
def jensen_shannon_divergence(diseases,val_diseases):
    diseases_counts, _ = np.histogram(diseases, bins=bins)
    H_counts, _ = np.histogram(Healthy, bins=bins)

    # Conversion en probabilités (somme = 1)
    diseases_probs = diseases_counts / diseases_counts.sum()
    H_probs = H_counts / H_counts.sum()

    # Jensen–Shannon divergence
    jsd = jensenshannon(diseases_probs, H_probs)
    print(f"Jensen–Shannon divergence = {jsd:.3f}")
    
    print(val_diseases)

#jensen_shannon_divergence(DR,val_DR)
#jensen_shannon_divergence(AMD,val_AMD)
#jensen_shannon_divergence(RVO,val_RVO)
#jensen_shannon_divergence(PM,val_PM)
#jensen_shannon_divergence(Uveitis,val_Uveitis)
#jensen_shannon_divergence(RD,val_RD)
#jensen_shannon_divergence(Healthy,val_Healthy)

#MAE vs JSD
def figure_age_mae_jsd():
    diseases_list = [Healthy,Uveitis,RD,PM,DR,RVO,AMD]
    diseases_names = ["Healthy", "Uveitis", "RD", "PM", "DR", "RVO", "AMD"]
    val_list = [val_Healthy,val_Uveitis,val_RD,val_PM,val_DR,val_RVO,val_AMD]
    JS_list = []

    for diseases in diseases_list:
        diseases_counts, _ = np.histogram(diseases, bins=bins)
        H_counts, _ = np.histogram(Healthy, bins=bins)

        # Conversion en probabilités (somme = 1)
        diseases_probs = diseases_counts / diseases_counts.sum()
        H_probs = H_counts / H_counts.sum()

        # Jensen–Shannon divergence
        jsd = jensenshannon(diseases_probs, H_probs)
        JS_list.append(round(float(jsd),3))
        
    #print(JS_list)
    x = np.arange(len(diseases_names))
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("Diseases")
    ax1.set_ylabel("MAE", color=color1)
    ax1.plot(x, val_list, marker="o", linestyle="-", color=color1, label="MAE")
    ax1.tick_params(axis="y", labelcolor=color1)


    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("JSD", color=color2)
    ax2.plot(x, JS_list, marker="s", linestyle="--", color=color2, label="JSD")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(diseases_names, rotation=20, ha="right")
    plt.title("MAE vs JSD by disease (relative to Healthy)")

    fig.tight_layout()
    plt.show()

#figure_age_mae_jsd()
def mae_interval(value_u, value_s):
    # ---- Indices de la tranche d'âge ----
    def take_index(lst):
        index = []
        for i in range(len(lst)):
            if value_s >= lst[i] >= value_u:
                index.append(i)
        return index

    dict_diseases_index = {
        "DR": take_index(DR),
        "AMD": take_index(AMD),
        "RVO": take_index(RVO),
        "PM": take_index(PM),
        "Uveitis": take_index(Uveitis),
        "RD": take_index(RD),
        "Healthy": take_index(Healthy)
    }

    # ---- Sous-listes des valeurs vraies ----
    dict_diseases_values = {
        disease: [lst[i] for i in indices]
        for disease, indices, lst in [
            ("DR", dict_diseases_index["DR"], DR),
            ("AMD", dict_diseases_index["AMD"], AMD),
            ("RVO", dict_diseases_index["RVO"], RVO),
            ("PM", dict_diseases_index["PM"], PM),
            ("Uveitis", dict_diseases_index["Uveitis"], Uveitis),
            ("RD", dict_diseases_index["RD"], RD),
            ("Healthy", dict_diseases_index["Healthy"], Healthy),
        ]
    }

    # ---- Lectures CSV (prédictions) ----
    import pandas as pd

    df = pd.read_csv(r"C:\Users\paulg\Desktop\DeepLearning\Ophthalmology_project\runs\predictions_Healthy.csv")
    pred_Healthy = df["pred_age"].tolist()

    df = pd.read_csv(r"C:\Users\paulg\Desktop\DeepLearning\Ophthalmology_project\runs\predictions_Uveitis.csv")
    pred_Uveitis = df["pred_age"].tolist()

    df = pd.read_csv(r"C:\Users\paulg\Desktop\DeepLearning\Ophthalmology_project\runs\predictions_RD.csv")
    pred_RD = df["pred_age"].tolist()

    df = pd.read_csv(r"C:\Users\paulg\Desktop\DeepLearning\Ophthalmology_project\runs\predictions_PM.csv")
    pred_PM = df["pred_age"].tolist()

    df = pd.read_csv(r"C:\Users\paulg\Desktop\DeepLearning\Ophthalmology_project\runs\predictions_DR.csv")
    pred_DR = df["pred_age"].tolist()

    df = pd.read_csv(r"C:\Users\paulg\Desktop\DeepLearning\Ophthalmology_project\runs\predictions_RVO.csv")
    pred_RVO = df["pred_age"].tolist()

    df = pd.read_csv(r"C:\Users\paulg\Desktop\DeepLearning\Ophthalmology_project\runs\predictions_AMD.csv")
    pred_AMD = df["pred_age"].tolist()

    # ---- Sous-listes des valeurs prédites (alignées sur indices) ----
    dict_diseases_values_predicted = {
        disease: [lst[i] for i in indices]
        for disease, indices, lst in [
            ("DR", dict_diseases_index["DR"], pred_DR),
            ("AMD", dict_diseases_index["AMD"], pred_AMD),
            ("RVO", dict_diseases_index["RVO"], pred_RVO),
            ("PM", dict_diseases_index["PM"], pred_PM),
            ("Uveitis", dict_diseases_index["Uveitis"], pred_Uveitis),
            ("RD", dict_diseases_index["RD"], pred_RD),
            ("Healthy", dict_diseases_index["Healthy"], pred_Healthy),
        ]
    }

    # ---- MAE par disease (arrondi 3 décimales, float natif) ----
    dict_diseases_mae = {}

    for disease in dict_diseases_values:
        true_vals = dict_diseases_values[disease]
        pred_vals = dict_diseases_values_predicted[disease]

        if len(true_vals) > 0:
            mae = sum(abs(t - p) for t, p in zip(true_vals, pred_vals)) / len(true_vals)
            mae = round(float(mae), 3)
        else:
            mae = None

        dict_diseases_mae[disease] = mae

    return dict_diseases_mae


#res_30_40 = mae_interval(30, 40)
#print(res_30_40)
#res_50_60 = mae_interval(50, 60)
#print(res_50_60)
#res_70_80 = mae_interval(70, 80)
#print(res_70_80)

import math

def plot_mae_by_age_bins():
    # --- Age bins: 0-10, 10-20, ..., 90-100 ---
    age_bins = [(i, i+10) for i in range(0, 100, 10)]  

    # --- Collect MAE for each bin using your existing function ---
    results_per_bin = [mae_interval(u, s) for (u, s) in age_bins]

    # --- X-axis: midpoint of each age bin ---
    x_vals = [(u + s) / 2 for (u, s) in age_bins]  

    # --- Diseases in desired order ---
    diseases = ["Healthy", "Uveitis", "RD", "PM", "DR", "RVO", "AMD"]

    # --- Plot one curve per disease ---
    plt.figure(figsize=(9, 5))
    for disease in diseases:
        y_vals = []
        for res in results_per_bin:
            mae = res.get(disease, None)
            y_vals.append(mae if mae is not None else float('nan'))
        plt.plot(x_vals, y_vals, marker='o', linewidth=2, label=disease)

    plt.title("MAE across age bins (10 years) — one curve per disease")
    plt.xlabel("Age (bin center)")
    plt.ylabel("MAE (years)")
    plt.xticks(x_vals, [f"{u}-{s}" for (u, s) in age_bins], rotation=45)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Disease")
    plt.tight_layout()
    plt.show()

#plot_mae_by_age_bins()