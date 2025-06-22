#here u can use my beautiful model
import torch
import numpy as np
from Data_Loader import dara_loaders
from Malevolence import MLP
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def predict_nk(input_vector, model_path, csv_for_config="C:\\Studia\\Progranmy\\AnalizaElipsometrii\\Scanner\\PreparedMaterials.csv"):
    psi, delta, lam, thickness, material = input_vector
    df = pd.read_csv(csv_for_config, sep="|")

    encoder = OneHotEncoder(sparse_output=False)
    material_encoded = encoder.fit_transform(df[["material"]])

    scaler = StandardScaler()
    X_raw = df[["psi_deg", "delta_deg", "lambda", "thickness"]].values
    X_full = np.concatenate([X_raw, material_encoded], axis=1)
    scaler.fit(X_full)

    base_vec = np.array([[psi, delta, lam, thickness]])
    material_vec = encoder.transform([[material]])  
    input_combined = np.concatenate([base_vec, material_vec], axis=1)

    input_scaled = scaler.transform(input_combined)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    input_dim = input_scaled.shape[1]
    model = MLP(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor).numpy()[0] 

    return {"n": output[0], "k": output[1]}


result = predict_nk(input_vector=[3.871, 1.5467, 250, 50, "SiO2"],model_path="C:\\Studia\\Progranmy\\AnalizaElipsometrii\\Scanner\\modelScanner.pt")
print("Predykcja:")
print(f"n = {result['n']:.4f}, k = {result['k']:.6f}")


