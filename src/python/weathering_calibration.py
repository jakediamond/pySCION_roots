import numpy as np

def log_r2(modelled_weathering, obs_weathering):

    RSS = sum((np.log(modelled_weathering) - np.log(obs_weathering))**2)
    TSS = sum((np.log(obs_weathering) - np.mean(np.log(obs_weathering)))**2)
    
    r2 = 1 - RSS/TSS
    
    return(r2)