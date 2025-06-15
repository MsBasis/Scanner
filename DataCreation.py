import pandas as pd 
import numpy as np 
from tmm import coh_tmm
from DataConfiguration import configList


all = []

def psi_delta(superior_n,wavelength,angle,n,d):
    n_list = [1.0,superior_n,n]
    d_list = [np.inf,d,np.inf]
    optic = coh_tmm('p',n_list,d_list,angle,wavelength)
    
    odbicie = optic['r']
    psi_tan = abs(odbicie)
    delta_rad = np.angle(odbicie)
    psi_deg = np.degrees(np.arctan(psi_tan))
    delta_deg = np.degrees(delta_rad) % 360
    return psi_deg, delta_deg


for mat in configList:
    df = pd.read_csv(mat['csv_path'])
    df.dropna(inplace=True)
    
    df['wl_nm'] = df['wl'].astype(float) * 1000
    df['n'] = df['n'].astype(float)
    df['k'] = df['k'].astype(float)
    
    for gragases in mat['gragas']:
        for _, row in df.iterrows():
            n = row['n']
            k = row['k']
            wl = row['wl_nm']
            superior_n = n +1j * k
            
            psi, delta = psi_delta(superior_n, wl, mat['kat'], mat['index'], gragases)
            all.append({
                'lambda' : wl,
                'psi_deg' : psi,
                'delta_deg' : delta,
                'n' : n,
                'k' : k,
                'thickness' : gragases,
                'material' : mat['material']
            })

df_out = pd.DataFrame(all)
df_out.to_csv("C:\\Studia\\Progranmy\\AnalizaElipsometrii\\Scanner\\PreparedMaterials.csv", index=False, sep='|')
















