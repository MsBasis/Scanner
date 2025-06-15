import numpy as np 
import pandas as pd 
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def dara_loaders(csv, test_size=0.2, batch_size=128):
    df = pd.read_csv(csv, sep='|')
    
    x_raw = df[['psi_deg','delta_deg','lambda','thickness','material']].copy()
    y = df[['n','k']].values.astype(np.float32)
    
    encoder = OneHotEncoder(sparse_output=False)
    material_encoded = encoder.fit_transform(x_raw[['material']])
    













