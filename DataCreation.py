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




















