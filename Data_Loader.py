import numpy as np 
import pandas as pd 
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class Mal_Dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(sefl.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


def dara_loaders(csv, test_size=0.2, batch_size=128):
    df = pd.read_csv(csv, sep='|')
    
    x_raw = df[['psi_deg','delta_deg','lambda','thickness','material']].copy()
    y = df[['n','k']].values.astype(np.float32)
    
    encoder = OneHotEncoder(sparse_output=False)
    material_encoded = encoder.fit_transform(x_raw[['material']])
    x_num = x_raw.drop(columns=['material']).values
    x_bigmom = np.concatenate([x_num, material_encoded], axis=1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_bigmom)
    
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=test_size, random_state=42)

    











