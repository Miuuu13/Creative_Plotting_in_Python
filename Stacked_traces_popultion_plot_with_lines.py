""" Tone onset as lines """


# %% Imports 

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime # for time stamp

# %% Functions to save/load .h5 as dictionary 
""" saving and loading of .h5 (laod is as df)"""

def save_hdf5(dict_data, folder="checkpoints", filename="events_dict.h5"):

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    with h5py.File(filepath, "w") as h5f:
        for key, array in dict_data.items():
            
            # Automatisch DataFrame zu NumPy umwandeln
            if isinstance(array, pd.DataFrame):
                array = array.to_numpy()
            
            if isinstance(array, (np.ndarray, list)):
                h5f.create_dataset(key, data=array, compression="gzip")
            else:
                raise ValueError(f"Can't save key '{key}': unsupported data type {type(array)}")

    print(f"Dictionary saved successfully at: {filepath}")



def load_hdf5_as_df(folder="checkpoints", filename="sessions_dict_same_session_length.h5"):
    filepath = os.path.join(folder, filename)
    loaded_dict = {}

    with h5py.File(filepath, "r") as h5f:
        for key in h5f.keys():
            loaded_dict[key] = pd.DataFrame(np.array(h5f[key]))

    print(f"Dictionary loaded (as DataFrames) successfully from: {filepath}")
    return loaded_dict

#%%

""" Load checkpoints for tone time frames """

#NOTE These checkpoint need to be saved inside folder named "checkpoints" in current working directory 

rp_csm = load_hdf5_as_df(folder="checkpoints", filename="rp_csm_200f_surrounding_tone_dict.h5")    #"rp_csm_checkpoint_0s_before_10s_after.h5")
#%%
rm_csm = load_hdf5_as_df(folder="checkpoints", filename="rm_csm_200f_surrounding_tone_dict.h5")
#%%
rp_csp = load_hdf5_as_df(folder="checkpoints", filename="rp_csp_200f_surrounding_tone_dict.h5")
#%%
rm_csp = load_hdf5_as_df(folder="checkpoints", filename="rm_csp_200f_surrounding_tone_dict.h5")
#%%
# %% CSP/CS+ Plotting (12 Tones) 
def plot_csp_traces(
    data_dict: dict, key: str, spacing:int=10, figsize=(12, 8),
    linewidth=0.5, fps=20
):
    if key not in data_dict:
        raise KeyError(f"Key '{key}' not found in dictionary.")

    df = data_dict[key]
    num_neurons = df.shape[1]
    num_frames = df.shape[0]
    time = np.arange(num_frames) / fps

    # tone formula
    tone_onsets_sec = [10 + i * 50 for i in range(12)]

    plt.figure(figsize=figsize)
    for i in range(num_neurons):
        trace = df.iloc[:, i].values
        offset_trace = trace + i * spacing
        plt.plot(time, offset_trace, linewidth=linewidth)

    for onset in tone_onsets_sec:
        plt.axvline(x=onset, linestyle='--', color='red', alpha=0.7)

    plt.title(f"Stacked Neuron Traces: {key}")
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index (stacked)")
    plt.yticks(np.arange(0, num_neurons * spacing, spacing), labels=np.arange(num_neurons))
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    filename = f"population_plot_{key}_csp_{datetime.now().strftime('%Y_%m_%d')}.svg"
    plt.savefig(filename, format='svg')
    plt.close()
    print(f"[CSP] Plot saved as: {filename}")

# %% CSM/CS- Plotting (4 Tones) 
def plot_csm_traces(
    data_dict:dict, key:str, spacing:int=10, figsize=(12, 8),
    linewidth=0.5, fps=20
):
    if key not in data_dict:
        raise KeyError(f"Key '{key}' not found in dictionary.")

    df = data_dict[key]
    num_neurons = df.shape[1]
    num_frames = df.shape[0]
    time = np.arange(num_frames) / fps

    # tone formula
    tone_onsets_sec = [10 + i * 50 for i in range(4)]

    plt.figure(figsize=figsize)
    for i in range(num_neurons):
        trace = df.iloc[:, i].values
        offset_trace = trace + i * spacing
        plt.plot(time, offset_trace, linewidth=linewidth)

    for onset in tone_onsets_sec:
        plt.axvline(x=onset, linestyle='--', color='red', alpha=0.7)

    plt.title(f"Stacked Neuron Traces: {key}")
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index (stacked)")
    plt.yticks(np.arange(0, num_neurons * spacing, spacing), labels=np.arange(num_neurons))
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    filename = f"population_plot_{key}_csm_{datetime.now().strftime('%Y_%m_%d')}.svg"
    plt.savefig(filename, format='svg')
    plt.close()
    print(f"[CSM] Plot saved as: {filename}")

# %% Examples: from session 4, same key for CS+ and CS-

#NOTE: each call currently saved the plot into current working directory, can be adjusted

plot_csp_traces(rp_csp, '936_A_s4_rp')

plot_csm_traces(rp_csm, '936_A_s4_rp')
