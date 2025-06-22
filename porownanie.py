import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from time import time
from tmm import coh_tmm

def psi_delta(superior_n, wavelength, angle, n, d):
    n_list = [1.0, superior_n, n]
    d_list = [np.inf, d, np.inf]
    optic = coh_tmm('p', n_list, d_list, angle, wavelength)
    odbicie = optic['r']
    psi_tan = abs(odbicie)
    delta_rad = np.angle(odbicie)
    psi_deg = np.degrees(np.arctan(psi_tan))
    delta_deg = np.degrees(delta_rad) % 360
    return psi_deg, delta_deg

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

df = pd.read_csv("C:\\Studia\\Progranmy\\AnalizaElipsometrii\\Scanner\\PreparedMaterials.csv", sep="|")
df = df.dropna()

grouped = df.groupby(['material', 'thickness'])

input_vectors = []
true_n = []
true_k = []
wavelengths = []
angles = []
superiors = []

for _, group in grouped:
    if len(group) < 8:
        continue
    group = group.iloc[:8]
    psi_delta_values = []
    for _, row in group.iterrows():
        psi_delta_values.extend([row['psi_deg'], row['delta_deg']])
    input_vectors.append(psi_delta_values)
    true_n.append(group['n'].values[0])
    true_k.append(group['k'].values[0])
    wavelengths.append(group['lambda'].values[0])
    angles.append(70)
    superiors.append(group['n'].values[0] + 1j * group['k'].values[0])
    if len(input_vectors) >= 50:
        break

X = np.array(input_vectors)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

model = MLP(input_dim=16)
model.load_state_dict(torch.load("C:\\Studia\\Progranmy\\AnalizaElipsometrii\\Scanner\\modelScanner3.pt", map_location="cpu"))
model.eval()

results = []
for i in range(len(X_norm)):
    x_tensor = torch.tensor([X_norm[i]], dtype=torch.float32)

    start_ml = time()
    with torch.no_grad():
        pred_ml = model(x_tensor).numpy().flatten()
    time_ml = time() - start_ml

    n_true = true_n[i]
    k_true = true_k[i]
    wl = wavelengths[i]
    angle = angles[i]
    superior_n = superiors[i]
    material = df.iloc[i]["material"]
    thickness = df[(df['material'] == material) & (df['thickness'] == df.iloc[i]["thickness"])]["thickness"].values[0]

    def cost(params):
        n, k = params
        if n <= 0 or k < 0:
            return 1e6
        try:
            psi_pred, delta_pred = psi_delta(n + 1j * k, wl, angle, superior_n, thickness)
            psi_true = df[(df['material'] == material) & (df['thickness'] == thickness)]["psi_deg"].mean()
            delta_true = df[(df['material'] == material) & (df['thickness'] == thickness)]["delta_deg"].mean()
            return (psi_pred - psi_true)**2 + (delta_pred - delta_true)**2
        except Exception:
            return 1e6

    start_classic = time()
    res = minimize(cost, x0=[1.5, 0.1], method='Nelder-Mead', options={'maxiter': 500})
    time_classic = time() - start_classic

    n_classic, k_classic = res.x if res.success else (np.nan, np.nan)

    results.append({
        "n_true": n_true,
        "k_true": k_true,
        "n_ML": pred_ml[0],
        "k_ML": pred_ml[1],
        "n_classic": n_classic,
        "k_classic": k_classic,
        "error_n_ML": abs(pred_ml[0] - n_true),
        "error_k_ML": abs(pred_ml[1] - k_true),
        "error_n_classic": abs(n_classic - n_true) if res.success else np.nan,
        "error_k_classic": abs(k_classic - k_true) if res.success else np.nan,
        "time_ML": time_ml,
        "time_classic": time_classic
    })

df_out = pd.DataFrame(results)
df_out.to_csv("C:\\Studia\\Progranmy\\AnalizaElipsometrii\\Scanner\\comparison_ML_vs_classic_fixed.csv", index=False)


