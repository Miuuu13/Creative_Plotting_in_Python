
#%%
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import scipy

from utils_io import load_hdf5 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

""" saving anf loading of .h5 (laod is as df)"""

def save_hdf5(dict_data, folder="checkpoints", filename="events_dict.h5"):
    import os
    import numpy as np
    import h5py
    import pandas as pd

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

""" start here - if checkpoints are in checkpoints folder"""

#%%  
""" Load checkpoints for tone time frames """

#%%
rm_csm = load_hdf5_as_df(folder="checkpoints", filename="rm_csm_200f_surrounding_tone_dict.h5")

#%%
rp_csm = load_hdf5_as_df(folder="checkpoints", filename="rp_csm_200f_surrounding_tone_dict.h5")


#%%
rm_csp = load_hdf5_as_df(folder="checkpoints", filename="rm_csp_200f_surrounding_tone_dict.h5")

#%%
rp_csp = load_hdf5_as_df(folder="checkpoints", filename="rp_csp_200f_surrounding_tone_dict.h5")
#%%
rp_csm
#%%
rm_csm
#%%
rp_csp
#%%
rm_csp

# %%


###################################

# POPULATION MEAN +- SEM (STD also tryed)

###################################
""" add tone exp shock and end line"""

""" add filtering batches, sessions; filtering checked"""

# #sourronding 200f = 10s
# CSM_TONE_STARTS = [200, 1200, 2200, 3200]  # CS-, 4 tones
# CSP_TONE_STARTS = [200, 1200, 2200, 3200, 4200, 5200, 6200, 7200, 8200, 9200, 10200, 11200]  # CS+, 12 tones
""" new version with filtering"""
# # Tone starts for CS+ and CS- (with 200 frames surrounding_before considered!)
# CSM_TONE_STARTS = np.array([200, 1200, 2200, 3200]) # CS-, 4 tones
# #CSP_TONE_STARTS = [200 + i * 1000 for i in range(12)]  #[200 + i * 1000 for i in range(12)]  # CS+, 12 tones
# CSP_TONE_STARTS = np.arange(200, 11300, 1000)
# CSP_TONE_EXP_SHOCK = np.arange(400, 11_700, 1000)
# CSP_TONE_END = np.arange(800, 12_100, 1000)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSM_TONE_STARTS = np.array([200, 1200, 2200, 3200])
CSP_TONE_STARTS = np.arange(200, 11300, 1000)

def filter_sessions(session_dict, include_batches=None, include_sessions=None, verbose=True):
    filtered_sessions = {}
    
    # Ensure batches/sessions are lists
    if include_batches is not None:
        include_batches = [str(b).upper() for b in include_batches]
    if include_sessions is not None:
        include_sessions = [str(s).lower() for s in include_sessions]
    
    for key, data in session_dict.items():
        parts = key.split('_')
        
        if len(parts) < 4:
            continue
        
        animal_id = '_'.join(parts[:-3])
        batch_label = parts[-3].upper()
        session_label = parts[-2].lower()
        type_label = parts[-1]

        batch_ok = (include_batches is None) or (batch_label in include_batches)
        session_ok = (include_sessions is None) or (session_label in include_sessions)

        if batch_ok and session_ok:
            filtered_sessions[key] = data
            if verbose:
                print(f"[INCLUDED] {key} -> batch={batch_label}, session={session_label}")
        else:
            if verbose:
                print(f"[EXCLUDED] {key} -> batch={batch_label}, session={session_label}, batch_ok={batch_ok}, session_ok={session_ok}")

    print(f"Filtered {len(filtered_sessions)} sessions out of {len(session_dict)}")
    return filtered_sessions


def plot_population_mean_trace_subplots(
    rp_dict,
    rm_dict,
    tone_type="CS+",
    frame_rate=20,
    include_batches=None,
    include_sessions=None,
    verbose=True
):
    if tone_type == "CS+":
        tone_starts = CSP_TONE_STARTS
    elif tone_type == "CS-":
        tone_starts = CSM_TONE_STARTS
    else:
        print(f"Unknown tone type: {tone_type}")
        return

    rp_filtered = filter_sessions(rp_dict, include_batches, include_sessions, verbose=verbose)
    rm_filtered = filter_sessions(rm_dict, include_batches, include_sessions, verbose=verbose)

    print("R+ FILTERED KEYS")
    for key in rp_filtered.keys():
        print(key)

    print("R- FILTERED KEYS")
    for key in rm_filtered.keys():
        print(key)

    def get_mean_traces(sessions_dict, label=''):
        traces_list = []
        for key, data in sessions_dict.items():
            if isinstance(data, pd.DataFrame):
                data = data.to_numpy()
            mean_trace = np.mean(data, axis=1)
            traces_list.append(mean_trace)
        if not traces_list:
            print(f"No data found in sessions_dict for {label}")
            return np.array([])
        min_length = min(len(trace) for trace in traces_list)
        traces_list = [trace[:min_length] for trace in traces_list]
        print(f"{label}: Found {len(traces_list)} traces, each of length {min_length}")
        return np.array(traces_list)

    rp_traces = get_mean_traces(rp_filtered, label='R+')
    rm_traces = get_mean_traces(rm_filtered, label='R-')

    if rp_traces.size == 0 or rm_traces.size == 0:
        print("Empty data in one of the groups, aborting plot")
        return
#     # Population mean and SEM (Standard Error of the Mean)     

    rp_mean = np.mean(rp_traces, axis=0)
    rp_sem = np.std(rp_traces, axis=0) / np.sqrt(rp_traces.shape[0]) #remove div. root for STD
#STD  np.std(rp_traces, axis=0)
    rm_mean = np.mean(rm_traces, axis=0)
    rm_sem = np.std(rm_traces, axis=0) / np.sqrt(rm_traces.shape[0])

    # ## error_type == 'std':
    # rp_error = np.std(rp_traces, axis=0)
    # rm_error = np.std(rm_traces, axis=0)

    time_axis = np.arange(len(rp_mean)) / frame_rate

    fig, axs = plt.subplots(1, 2, figsize=(24, 6), sharey=True)

    ax = axs[0]
    ax.plot(time_axis, rp_mean, label='R+', color='blue')
    ax.fill_between(time_axis, rp_mean - rp_sem, rp_mean + rp_sem, color='blue', alpha=0.3)
    ax.set_title(f'{tone_type} - R+ (n={rp_traces.shape[0]})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Fluorescence (a.u.)')
    for tone_start in tone_starts:
        tone_time = tone_start / frame_rate
        ax.axvline(x=tone_time, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=tone_time + 10, color='orange', linestyle='--', linewidth=1)
        ax.axvline(x=tone_time + 30, color='cyan', linestyle='--', linewidth=1)
    ax.legend()
    ax.grid(True)

    ax = axs[1]
    ax.plot(time_axis, rm_mean, label='R-', color='red')
    ax.fill_between(time_axis, rm_mean - rm_sem, rm_mean + rm_sem, color='red', alpha=0.3)
    ax.set_title(f'{tone_type} - R- (n={rm_traces.shape[0]})')
    ax.set_xlabel('Time (s)')
    for tone_start in tone_starts:
        tone_time = tone_start / frame_rate
        ax.axvline(x=tone_time, color='red', linestyle='--', linewidth=1)
        ax.axvline(x=tone_time + 10, color='orange', linestyle='--', linewidth=1)
        ax.axvline(x=tone_time + 30, color='cyan', linestyle='--', linewidth=1)
    ax.legend()
    ax.grid(True)

    batch_text = f"Batches: {include_batches}" if include_batches else "All Batches"
    session_text = f"Sessions: {include_sessions}" if include_sessions else ""
    combined_text = f"{batch_text} | {session_text}" if session_text else batch_text

    plt.suptitle(f'Mean Fluorescence Over Time ({tone_type}) | {combined_text}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print(f"R+ sessions: {rp_traces.shape[0]} animals, {rp_traces.shape[1]} frames per trace")
    print(f"R- sessions: {rm_traces.shape[0]} animals, {rm_traces.shape[1]} frames per trace")

# %%
plot_population_mean_trace_subplots(
    rp_dict=rp_csp,
    rm_dict=rm_csp,
    tone_type="CS+",
    frame_rate=20,
    include_batches=['A', 'B'],
    include_sessions=['s4'],
    verbose=True
)

#%%
plot_population_mean_trace_subplots(
    rp_dict=rp_csm,
    rm_dict=rm_csm,
    tone_type="CS-",
    frame_rate=20,
    include_batches=['A', 'B'],
    include_sessions=['s4'],
    verbose=True
)

# %%
rp_csm.keys()
# %%
# %%

"""  Function to Stack Tones into a 2D Matrix - without np.arrange!!!!"""

def stack_tone_traces_without_arrange(sessions_dict, n, verbose=True):
    """
    Stack each trace in a sessions_dict into n tones as rows in a 2D matrix.

    Args:
        sessions_dict (dict): Input dict with key → data (numpy arrays or dataframes)
        n (int): Number of tones to split into
        verbose (bool): Print summary info
    
    Returns:
        stacked_dict (dict): New dict where each session is a (n, frames_per_tone) matrix
    """
    stacked_dict = {}
    
    for key, data in sessions_dict.items():
        # Convert DataFrame to numpy if necessary(?)
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        # Mean across neurons if 2D (time x neurons)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        
        total_frames = len(data)
        frames_per_tone = total_frames // n
        
        # Split trace into n tones (rows of the matrix); n=4 for cs-, 12 for cs+
        stacked_matrix = np.zeros((n, frames_per_tone))
        
        for i in range(n):
            start_idx = i * frames_per_tone
            end_idx = start_idx + frames_per_tone
            stacked_matrix[i, :] = data[start_idx:end_idx]
        
        stacked_dict[key] = stacked_matrix
        
        if verbose:
            print(f"[{key}] Original trace length: {total_frames}, Tones: {n}, Frames per tone: {frames_per_tone}")
    
    print(f"- Stacked {len(stacked_dict)} sessions into tone matrices with {n} tones each.")
    return stacked_dict



""" Function ussing arrange:"""


def stack_tone_traces(sessions_dict, n, verbose=True):
    """
    Stack each trace in a sessions_dict into n tones as rows in a 2D matrix.

    Args:
        sessions_dict (dict): Input dict with key → data (numpy arrays or dataframes)
        n (int): Number of tones to split into
        verbose (bool): Print summary info
    
    Returns:
        stacked_dict (dict): New dict where each session is a (n, frames_per_tone) matrix
    """
    stacked_dict = {}
    
    for key, data in sessions_dict.items():
        # Convert DataFrame to numpy if necessary(?)
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        # Mean across neurons if 2D (time x neurons)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        
        total_frames = len(data)
        frames_per_tone = total_frames // n
        
        # Generate start and end indices using np.arange()
        indices = np.arange(0, total_frames + 1, frames_per_tone)
        
        # Split trace into n tones (rows of the matrix); n=4 for cs-, 12 for cs+
        stacked_matrix = np.zeros((n, frames_per_tone))
        
        for i in range(n):
            stacked_matrix[i, :] = data[indices[i]:indices[i+1]]
        
        stacked_dict[key] = stacked_matrix
        
        if verbose:
            print(f"[{key}] Original trace length: {total_frames}, Tones: {n}, Frames per tone: {frames_per_tone}")
    
    print(f"- Stacked {len(stacked_dict)} sessions into tone matrices with {n} tones each.")
    return stacked_dict



#%%


#%%
"""actual 2d stacking"""
#%%
rp_csm_stacked = stack_tone_traces(rp_csm, n=4)
rm_csm_stacked = stack_tone_traces(rm_csm, n=4)
rp_csp_stacked = stack_tone_traces(rp_csp, n=12)
rm_csp_stacked = stack_tone_traces(rm_csp, n=12)

#%%

for key in rp_csm_stacked: print(rp_csm_stacked[key].shape)
# #%%
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_rp_vs_rm_for_cs_types(
#     rp_csm_stacked,
#     rm_csm_stacked,
#     rp_csp_stacked,
#     rm_csp_stacked,
#     frame_rate=20
# ):
#     """
#     Plots R+ vs R- for CS- and CS+ separately.
    
#     Args:
#         rp_csm_stacked (dict): Stacked CS- R+ dict
#         rm_csm_stacked (dict): Stacked CS- R- dict
#         rp_csp_stacked (dict): Stacked CS+ R+ dict
#         rm_csp_stacked (dict): Stacked CS+ R- dict
#         frame_rate (int): Frames per second for time axis
#     """
    
#     # csm
#     rp_csm_matrix = np.vstack([matrix for matrix in rp_csm_stacked.values()])
#     rm_csm_matrix = np.vstack([matrix for matrix in rm_csm_stacked.values()])
    
#     rp_csm_mean = np.mean(rp_csm_matrix, axis=0)
#     rp_csm_sem  = np.std(rp_csm_matrix, axis=0) / np.sqrt(rp_csm_matrix.shape[0])
    
#     rm_csm_mean = np.mean(rm_csm_matrix, axis=0)
#     rm_csm_sem  = np.std(rm_csm_matrix, axis=0) / np.sqrt(rm_csm_matrix.shape[0])
    
#     #plotting csm
#     time_axis_csm = np.arange(rp_csm_mean.shape[0]) / frame_rate

#     alpha_value = 0.5
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_axis_csm, rp_csm_mean, label=f'R+ CS- ({rp_csm_matrix.shape[0]} tones)', color='green')
#     plt.fill_between(time_axis_csm, rp_csm_mean - rp_csm_sem, rp_csm_mean + rp_csm_sem, color='green', alpha=alpha_value)
    
#     plt.plot(time_axis_csm, rm_csm_mean, label=f'R- CS- ({rm_csm_matrix.shape[0]} tones)', color='orange')
#     plt.fill_between(time_axis_csm, rm_csm_mean - rm_csm_sem, rm_csm_mean + rm_csm_sem, color='orange', alpha=alpha_value)
    
#     plt.title('R+ vs R- for CS-')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Mean Fluorescence (a.u.)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
#     #csp
#     rp_csp_matrix = np.vstack([matrix for matrix in rp_csp_stacked.values()])
#     rm_csp_matrix = np.vstack([matrix for matrix in rm_csp_stacked.values()])
    
#     rp_csp_mean = np.mean(rp_csp_matrix, axis=0)
#     rp_csp_sem  = np.std(rp_csp_matrix, axis=0) / np.sqrt(rp_csp_matrix.shape[0])
    
#     rm_csp_mean = np.mean(rm_csp_matrix, axis=0)
#     rm_csp_sem  = np.std(rm_csp_matrix, axis=0) / np.sqrt(rm_csp_matrix.shape[0])
    
#     #csp
#     time_axis_csp = np.arange(rp_csp_mean.shape[0]) / frame_rate
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_axis_csp, rp_csp_mean, label=f'R+ CS+ ({rp_csp_matrix.shape[0]} tones)', color='green')
#     plt.fill_between(time_axis_csp, rp_csp_mean - rp_csp_sem, rp_csp_mean + rp_csp_sem, color='green', alpha=alpha_value)
    
#     plt.plot(time_axis_csp, rm_csp_mean, label=f'R- CS+ ({rm_csp_matrix.shape[0]} tones)', color='orange')
#     plt.fill_between(time_axis_csp, rm_csp_mean - rm_csp_sem, rm_csp_mean + rm_csp_sem, color='orange', alpha= alpha_value)
    
#     plt.title('R+ vs R- for CS+')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Activity (mean +- sem)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    


#     print(f"CS-: R+ tones = {rp_csm_matrix.shape[0]}, R- tones = {rm_csm_matrix.shape[0]}")
#     print(f"CS+: R+ tones = {rp_csp_matrix.shape[0]}, R- tones = {rm_csp_matrix.shape[0]}")

#%%

# plot_rp_vs_rm_for_cs_types(
#     rp_csm_stacked=rp_csm_stacked,
#     rm_csm_stacked=rm_csm_stacked,
#     rp_csp_stacked=rp_csp_stacked,
#     rm_csp_stacked=rm_csp_stacked,
#     frame_rate=20
# )
#%%


import matplotlib.pyplot as plt
import numpy as np

def plot_rp_vs_rm_for_cs_types(
    rp_csm_stacked,
    rm_csm_stacked,
    rp_csp_stacked,
    rm_csp_stacked,
    batch_list=None,
    session=None,
    frame_rate=20,
    verbose=True
):
    """
    Plots R+ vs R- for CS- and CS+ separately, with optional filtering by batch_list and session.

    Args:
        rp_csm_stacked (dict): Stacked CS- R+ dict
        rm_csm_stacked (dict): Stacked CS- R- dict
        rp_csp_stacked (dict): Stacked CS+ R+ dict
        rm_csp_stacked (dict): Stacked CS+ R- dict
        batch_list (list): List of batch labels to filter, e.g. ['A', 'B']
        session (str): Session label to filter, e.g. 's4'
        frame_rate (int): Frames per second for time axis
        verbose (bool): Print included/excluded keys
    """

    # Prepare filtering function
    def filter_dict(stacked_dict, batch_list=None, session=None):
        filtered_dict = {}
        if batch_list is not None:
            batch_list = [b.upper() for b in batch_list]
        if session is not None:
            session = session.lower()

        for key, matrix in stacked_dict.items():
            parts = key.split('_')
            if len(parts) < 4:
                continue
            batch_label = parts[-3].upper()
            session_label = parts[-2].lower()

            batch_ok = (batch_list is None) or (batch_label in batch_list)
            session_ok = (session is None) or (session_label == session)

            if batch_ok and session_ok:
                filtered_dict[key] = matrix
                if verbose:
                    print(f"[INCLUDED] {key} (batch={batch_label}, session={session_label})")
            elif verbose:
                print(f"[EXCLUDED] {key} (batch={batch_label}, session={session_label})")

        print(f"✅ Filtered {len(filtered_dict)} sessions out of {len(stacked_dict)}")
        return filtered_dict

    # Filter all dictionaries
    rp_csm_filtered = filter_dict(rp_csm_stacked, batch_list=batch_list, session=session)
    rm_csm_filtered = filter_dict(rm_csm_stacked, batch_list=batch_list, session=session)
    rp_csp_filtered = filter_dict(rp_csp_stacked, batch_list=batch_list, session=session)
    rm_csp_filtered = filter_dict(rm_csp_stacked, batch_list=batch_list, session=session)

    alpha_value = 0.5


    # =========================
    if rp_csm_filtered and rm_csm_filtered:
        rp_csm_matrix = np.vstack([matrix for matrix in rp_csm_filtered.values()])
        rm_csm_matrix = np.vstack([matrix for matrix in rm_csm_filtered.values()])

        rp_csm_mean = np.mean(rp_csm_matrix, axis=0)
        rp_csm_sem  = np.std(rp_csm_matrix, axis=0) / np.sqrt(rp_csm_matrix.shape[0])

        rm_csm_mean = np.mean(rm_csm_matrix, axis=0)
        rm_csm_sem  = np.std(rm_csm_matrix, axis=0) / np.sqrt(rm_csm_matrix.shape[0])

        time_axis_csm = np.arange(rp_csm_mean.shape[0]) / frame_rate

        rp_csm_mean = scipy.stats.zscore(rp_csm_mean)
        
        rm_csm_mean = scipy.stats.zscore(rm_csm_mean)


        plt.figure(figsize=(10, 6))
        plt.plot(time_axis_csm, rp_csm_mean, label=f'R+ CS- ({rp_csm_matrix.shape[0]} tones)', color='green')
        plt.fill_between(time_axis_csm, rp_csm_mean - rp_csm_sem, rp_csm_mean + rp_csm_sem, color='green', alpha=alpha_value)

        plt.plot(time_axis_csm, rm_csm_mean, label=f'R- CS- ({rm_csm_matrix.shape[0]} tones)', color='orange')
        plt.fill_between(time_axis_csm, rm_csm_mean - rm_csm_sem, rm_csm_mean + rm_csm_sem, color='orange', alpha=alpha_value)

        plt.title(f'R+ vs R- for CS- | Batches: {batch_list if batch_list else "ALL"} | Session: {session if session else "ALL"}')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Fluorescence (a.u.)')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("⚠️ No data to plot for CS-.")


    # =========================
    if rp_csp_filtered and rm_csp_filtered:
        rp_csp_matrix = np.vstack([matrix for matrix in rp_csp_filtered.values()])
        rm_csp_matrix = np.vstack([matrix for matrix in rm_csp_filtered.values()])

        rp_csp_mean = np.mean(rp_csp_matrix, axis=0)
        rp_csp_sem  = np.std(rp_csp_matrix, axis=0) / np.sqrt(rp_csp_matrix.shape[0])

        rm_csp_mean = np.mean(rm_csp_matrix, axis=0)
        rm_csp_sem  = np.std(rm_csp_matrix, axis=0) / np.sqrt(rm_csp_matrix.shape[0])

        rp_csp_mean = scipy.stats.zscore(rp_csp_mean)         
        rm_csp_mean = scipy.stats.zscore(rm_csp_mean)

        time_axis_csp = np.arange(rp_csp_mean.shape[0]) / frame_rate

        plt.figure(figsize=(10, 6))
        plt.plot(time_axis_csp, rp_csp_mean, label=f'R+ CS+ ({rp_csp_matrix.shape[0]} tones)', color='green')
        plt.fill_between(time_axis_csp, rp_csp_mean - rp_csp_sem, rp_csp_mean + rp_csp_sem, color='green', alpha=alpha_value)

        plt.plot(time_axis_csp, rm_csp_mean, label=f'R- CS+ ({rm_csp_matrix.shape[0]} tones)', color='orange')
        plt.fill_between(time_axis_csp, rm_csp_mean - rm_csp_sem, rm_csp_mean + rm_csp_sem, color='orange', alpha=alpha_value)

        plt.title(f'R+ vs R- for CS+ | Batches: {batch_list if batch_list else "ALL"} | Session: {session if session else "ALL"}')
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron activit - mean +- sem ')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("⚠️ No data to plot for CS+.")

    if rp_csm_filtered:
        print(f"CS-: R+ tones = {len(rp_csm_filtered) * rp_csm_filtered[list(rp_csm_filtered.keys())[0]].shape[0]}, R- tones = {len(rm_csm_filtered) * rm_csm_filtered[list(rm_csm_filtered.keys())[0]].shape[0]}")
    if rp_csp_filtered:
        print(f"CS+: R+ tones = {len(rp_csp_filtered) * rp_csp_filtered[list(rp_csp_filtered.keys())[0]].shape[0]}, R- tones = {len(rm_csp_filtered) * rm_csp_filtered[list(rm_csp_filtered.keys())[0]].shape[0]}")


#%%
"""" for plotting all """
plot_rp_vs_rm_for_cs_types(
    rp_csm_stacked,
    rm_csm_stacked,
    rp_csp_stacked,
    rm_csp_stacked,
    batch_list=None,
    session=None
)

#%%
# only filtered
plot_rp_vs_rm_for_cs_types(
    rp_csm_stacked,
    rm_csm_stacked,
    rp_csp_stacked,
    rm_csp_stacked,
    batch_list=['B'],
    session='s4'
)
#%%
plot_rp_vs_rm_for_cs_types(
    rp_csm_stacked,
    rm_csm_stacked,
    rp_csp_stacked,
    rm_csp_stacked,
    batch_list=['C', 'D'],
    session='s4'
)
#%%
plot_rp_vs_rm_for_cs_types(
    rp_csm_stacked,
    rm_csm_stacked,
    rp_csp_stacked,
    rm_csp_stacked,
    batch_list=['E', 'F'],
    session='s4'
)


#%%

"""" Inspect session 3"""

#%%
all_sessions_dict = load_hdf5_as_df(folder="checkpoints", filename="sessions_dict_same_min_length_checkpoint.h5")

##Filter s3


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
#totnes
CSM_TONE_STARTS = np.array([200, 1200, 2200, 3200])
CSP_TONE_STARTS = np.arange(200, 11300, 1000)
## Filter session 3 out
def extract_session_3(sessions_dict, deepcopy=True):
    extracted = {}
    for key, data in sessions_dict.items():
        parts = key.split('_')
        if len(parts) < 4:
            continue
        session_label = parts[-2].lower()
        if session_label == 's3':
            extracted[key] = copy.deepcopy(data) if deepcopy else data
    print(f"Extracted {len(extracted)} sessions for s3")
    return extracted

#Trim session 3 data to the region of interest
def trim_session_3_data(session_data, frame_rate=20, verbose=True):
    trimmed_sessions = {}
    for key, data in session_data.items():
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        
        # Session 3 total length is 3600 frames.
        # You want:
        # - 10 sec before CSP -> 10 * 20 = 200 frames before CSP
        # - CSP starts after habituation (180 sec = 3600 frames)
        #    → but in your description, total session is 3600 frames?
        # Let's assume CSP starts around 3600, but confirm your numbers!

        # Example values:
        habituation_frames = 3600
        csp_start = habituation_frames               # Frame 3600
        csp_end = csp_start + 20                    # 1-sec CSP tone (20 frames)
        csm_start = csp_end + 1200                  # 60 sec later (pause) + 10 sec CSM
        
        # Extract frames from 10 sec before CSP to 10 sec after CSM
        start_frame = csp_start - 200
        end_frame = csm_start + 200

        trimmed_data = data[start_frame:end_frame, :]  # time x neurons
        trimmed_sessions[key] = trimmed_data

        if verbose:
            print(f"[{key}] Trimmed from {start_frame} to {end_frame} → shape {trimmed_data.shape}")

    return trimmed_sessions
#gruop r+ r-
def split_rp_rm(sessions_dict):
    rp_dict = {}
    rm_dict = {}
    for key, data in sessions_dict.items():
        if '_rp' in key:
            rp_dict[key] = data
        elif '_rm' in key:
            rm_dict[key] = data
        else:
            print(f"⚠️ Unknown group in key {key}")
    print(f"Split into R+ ({len(rp_dict)}), R- ({len(rm_dict)})")
    return rp_dict, rm_dict


def plot_trimmed_session_3(rp_dict, rm_dict, frame_rate=20):
    def get_mean_traces(sessions_dict, label=''):
        traces_list = []
        for key, data in sessions_dict.items():
            mean_trace = np.mean(data, axis=1)
            traces_list.append(mean_trace)
        if not traces_list:
            print(f"No data for {label}")
            return np.array([])
        min_length = min(len(trace) for trace in traces_list)
        traces_list = [trace[:min_length] for trace in traces_list]
        print(f"{label}: {len(traces_list)} traces, length {min_length}")
        return np.array(traces_list)

    rp_traces = get_mean_traces(rp_dict, label='R+')
    rm_traces = get_mean_traces(rm_dict, label='R-')

    if rp_traces.size == 0 or rm_traces.size == 0:
        print("Nothing to plot")
        return

    rp_mean = np.mean(rp_traces, axis=0)
    rm_mean = np.mean(rm_traces, axis=0)

    rp_sem = np.std(rp_traces, axis=0) / np.sqrt(rp_traces.shape[0])
    rm_sem = np.std(rm_traces, axis=0) / np.sqrt(rm_traces.shape[0])

    time_axis = np.arange(len(rp_mean)) / frame_rate

    fig, axs = plt.subplots(1, 2, figsize=(24, 6), sharey=True)

    # R+
    ax = axs[0]
    ax.plot(time_axis, rp_mean, label='R+', color='blue')
    ax.fill_between(time_axis, rp_mean - rp_sem, rp_mean + rp_sem, color='blue', alpha=0.3)
    ax.set_title(f'Session 3 - R+ (n={rp_traces.shape[0]})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Fluorescence (a.u.)')
    
    # CSP and CSM marker lines
    ax.axvline(x=10, color='black', linestyle='--', label='CSP Start')        # 10 sec after trim start
    ax.axvline(x=11, color='cyan', linestyle='--', label='CSP End')          # CSP + 1 sec
    ax.axvline(x=71, color='orange', linestyle='--', label='CSM Start')      # 60 sec later + 1 sec CSP = 71 sec

    ax.legend()
    ax.grid(True)

    # R-
    ax = axs[1]
    ax.plot(time_axis, rm_mean, label='R-', color='red')
    ax.fill_between(time_axis, rm_mean - rm_sem, rm_mean + rm_sem, color='red', alpha=0.3)
    ax.set_title(f'Session 3 - R- (n={rm_traces.shape[0]})')
    ax.set_xlabel('Time (s)')

    ax.axvline(x=10, color='black', linestyle='--', label='CSP Start')
    ax.axvline(x=11, color='cyan', linestyle='--', label='CSP End')
    ax.axvline(x=71, color='orange', linestyle='--', label='CSM Start')

    ax.legend()
    ax.grid(True)

    plt.suptitle('Session 3 - Trimmed Plot')
    plt.tight_layout()
    plt.show()

    print(f"R+ animals: {rp_traces.shape[0]}, R- animals: {rm_traces.shape[0]}")



#%%  extract session 3
session3_dict = extract_session_3(all_sessions_dict)

trimmed_session3_dict = trim_session_3_data(session3_dict)


rp_s3, rm_s3 = split_rp_rm(trimmed_session3_dict)

plot_trimmed_session_3(rp_s3, rm_s3)

# %%



################

#%%
#%%


import matplotlib.pyplot as plt
import numpy as np

def plot_rp_vs_rm_for_cs_types(
    rp_csm_stacked,
    rm_csm_stacked,
    rp_csp_stacked,
    rm_csp_stacked,
    batch_list=None,
    session=None,
    frame_rate=20,
    verbose=True
):
    """
    Plots R+ vs R- for CS- and CS+ separately, with optional filtering by batch_list and session.

    Args:
        rp_csm_stacked (dict): Stacked CS- R+ dict
        rm_csm_stacked (dict): Stacked CS- R- dict
        rp_csp_stacked (dict): Stacked CS+ R+ dict
        rm_csp_stacked (dict): Stacked CS+ R- dict
        batch_list (list): List of batch labels to filter, e.g. ['A', 'B']
        session (str): Session label to filter, e.g. 's4'
        frame_rate (int): Frames per second for time axis
        verbose (bool): Print included/excluded keys
    """

    # Prepare filtering function
    def filter_dict(stacked_dict, batch_list=None, session=None):
        filtered_dict = {}
        if batch_list is not None:
            batch_list = [b.upper() for b in batch_list]
        if session is not None:
            session = session.lower()

        for key, matrix in stacked_dict.items():
            parts = key.split('_')
            if len(parts) < 4:
                continue
            batch_label = parts[-3].upper()
            session_label = parts[-2].lower()

            batch_ok = (batch_list is None) or (batch_label in batch_list)
            session_ok = (session is None) or (session_label == session)

            if batch_ok and session_ok:
                filtered_dict[key] = matrix
                if verbose:
                    print(f"[INCLUDED] {key} (batch={batch_label}, session={session_label})")
            elif verbose:
                print(f"[EXCLUDED] {key} (batch={batch_label}, session={session_label})")

        print(f" Filtered {len(filtered_dict)} sessions out of {len(stacked_dict)}")
        return filtered_dict

    # Filter all dictionaries
    rp_csm_filtered = filter_dict(rp_csm_stacked, batch_list=batch_list, session=session)
    rm_csm_filtered = filter_dict(rm_csm_stacked, batch_list=batch_list, session=session)
    rp_csp_filtered = filter_dict(rp_csp_stacked, batch_list=batch_list, session=session)
    rm_csp_filtered = filter_dict(rm_csp_stacked, batch_list=batch_list, session=session)

    alpha_value = 0.5


    




    # =========================
    if rp_csm_filtered and rm_csm_filtered:
        rp_csm_matrix = np.vstack([matrix for matrix in rp_csm_filtered.values()])
        rm_csm_matrix = np.vstack([matrix for matrix in rm_csm_filtered.values()])

        rp_csm_mean = np.mean(rp_csm_matrix, axis=0)
        rp_csm_sem  = np.std(rp_csm_matrix, axis=0) / np.sqrt(rp_csm_matrix.shape[0])

        rm_csm_mean = np.mean(rm_csm_matrix, axis=0)
        rm_csm_sem  = np.std(rm_csm_matrix, axis=0) / np.sqrt(rm_csm_matrix.shape[0])

        time_axis_csm = np.arange(rp_csm_mean.shape[0]) / frame_rate

        #rp_csm_mean = scipy.stats.zscore(rp_csm_mean)
        
        #rm_csm_mean = scipy.stats.zscore(rm_csm_mean)




    # =========================
    if rp_csp_filtered and rm_csp_filtered:
        rp_csp_matrix = np.vstack([matrix for matrix in rp_csp_filtered.values()])
        rm_csp_matrix = np.vstack([matrix for matrix in rm_csp_filtered.values()])

        rp_csp_mean = np.mean(rp_csp_matrix, axis=0)
        rp_csp_sem  = np.std(rp_csp_matrix, axis=0) / np.sqrt(rp_csp_matrix.shape[0])

        rm_csp_mean = np.mean(rm_csp_matrix, axis=0)
        rm_csp_sem  = np.std(rm_csp_matrix, axis=0) / np.sqrt(rm_csp_matrix.shape[0])

        plotDict = {"R+": rp_csp_matrix, "R-":rm_csp_matrix}
        plotDict = {"R+": rp_csp_matrix.reshape(12,rp_csp_matrix.shape[0]//12,1000), "R-":rm_csp_matrix.reshape(12,rm_csp_matrix.shape[0]//12,1000)}

        #rp_csp_mean = scipy.stats.zscore(rp_csp_mean)         
        #rm_csp_mean = scipy.stats.zscore(rm_csp_mean)
    for i, cond in enumerate(["R+", "R-"]):
        nAnimal = plotDict[cond].shape[1]
        for i in range(nAnimal):
            #plotDict[cond][:, i, :] = scipy.stats.zscore(plotDict[cond][:, i, :])
            plotDict[cond][:, i, :] = plotDict[cond][:, i, :] - plotDict[cond][:, i, :].mean()
        plotDict[cond] = plotDict[cond].reshape([12*nAnimal, 1000])


    minMax = []
    print(f"BEFORE")
    for i, cond in enumerate(["R+", "R-"]):
        key = cond
        xy = plotDict[key]
        print(f"HERE: {xy.shape}")
        #xy = xy[:,1000:]
        minMax.append([xy.min(), xy.max()])
    print(minMax)
    minMax = [np.min(minMax, axis=0)[0], np.max(minMax, axis=0)[1]]
    print(minMax)

    fig, axMatrix = plt.subplots(1,2, figsize=(8,3), tight_layout=True,dpi=200)
    axList = axMatrix.flatten()

    before_tone = 10

    for i, cond in enumerate(["R+", "R-"]):
        ax = axList[i]
        key = cond
        xy = plotDict[key]
        
        imshow_plot = ax.imshow(xy, vmin=minMax[0], vmax=minMax[1], aspect="auto",interpolation='none')
        fps = 20
        ax.set_xlim(0,50*fps)
        xticks = np.arange(0,1001,1000)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks//fps-10)# minus 15 to align to tone onset
        #x.set_yticks(np.arange(4))  
        #ax.set_yticklabels([f"Ext{tick}" for tick in (np.arange(4)+1)])  
        ax.axvline(before_tone*fps, c="orange")
        ax.axvline((before_tone+10)*fps, c="orange")
        ax.axvline((before_tone+30)*fps, c="orange")
        ax.set_title(cond)
        ax.set_xlabel("Time since tone onset [sec]")
        ax.set_ylabel("Tones")
        for i in range(xy.shape[0]//12):
            ax.axhline(i*12, color="w")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.99, 0.195, 0.015, 0.68])
    fig.colorbar(imshow_plot, cax=cbar_ax, label="Neuronal activity (z-score)")
    #fig.colorbar(imshow_plot, ax=ax, location='right', anchor=(0, 0.3), label="Neuronal activity (z-score)")#, shrink=0.7)

    #fig.savefig(f"luminanceHeatmap_Rpm_sameScale_test.svg")
   

# %%
plot_rp_vs_rm_for_cs_types(
    rp_csm_stacked,
    rm_csm_stacked,
    rp_csp_stacked,
    rm_csp_stacked,
    batch_list=['A','B'],
    session="s4",
    frame_rate=20,
    verbose=True
)
# %%
""" add zscore"""


import matplotlib.pyplot as plt
import numpy as np

def plot_rp_vs_rm_for_cs_types(
    rp_csm_stacked,
    rm_csm_stacked,
    rp_csp_stacked,
    rm_csp_stacked,
    batch_list=None,
    session=None,
    frame_rate=20,
    verbose=True
):
    """
    Plots R+ vs R- for CS- and CS+ separately, with optional filtering by batch_list and session.

    Args:
        rp_csm_stacked (dict): Stacked CS- R+ dict
        rm_csm_stacked (dict): Stacked CS- R- dict
        rp_csp_stacked (dict): Stacked CS+ R+ dict
        rm_csp_stacked (dict): Stacked CS+ R- dict
        batch_list (list): List of batch labels to filter, e.g. ['A', 'B']
        session (str): Session label to filter, e.g. 's4'
        frame_rate (int): Frames per second for time axis
        verbose (bool): Print included/excluded keys
    """

    # Prepare filtering function
    def filter_dict(stacked_dict, batch_list=None, session=None):
        filtered_dict = {}
        if batch_list is not None:
            batch_list = [b.upper() for b in batch_list]
        if session is not None:
            session = session.lower()

        for key, matrix in stacked_dict.items():
            parts = key.split('_')
            if len(parts) < 4:
                continue
            batch_label = parts[-3].upper()
            session_label = parts[-2].lower()

            batch_ok = (batch_list is None) or (batch_label in batch_list)
            session_ok = (session is None) or (session_label == session)

            if batch_ok and session_ok:
                filtered_dict[key] = matrix
                if verbose:
                    print(f"[INCLUDED] {key} (batch={batch_label}, session={session_label})")
            elif verbose:
                print(f"[EXCLUDED] {key} (batch={batch_label}, session={session_label})")

        print(f" Filtered {len(filtered_dict)} sessions out of {len(stacked_dict)}")
        return filtered_dict

    # Filter all dictionaries
    rp_csm_filtered = filter_dict(rp_csm_stacked, batch_list=batch_list, session=session)
    rm_csm_filtered = filter_dict(rm_csm_stacked, batch_list=batch_list, session=session)
    rp_csp_filtered = filter_dict(rp_csp_stacked, batch_list=batch_list, session=session)
    rm_csp_filtered = filter_dict(rm_csp_stacked, batch_list=batch_list, session=session)

    alpha_value = 0.5


    




    # =========================
    if rp_csm_filtered and rm_csm_filtered:
        rp_csm_matrix = np.vstack([matrix for matrix in rp_csm_filtered.values()])
        rm_csm_matrix = np.vstack([matrix for matrix in rm_csm_filtered.values()])

        rp_csm_mean = np.mean(rp_csm_matrix, axis=0)
        rp_csm_sem  = np.std(rp_csm_matrix, axis=0) / np.sqrt(rp_csm_matrix.shape[0])

        rm_csm_mean = np.mean(rm_csm_matrix, axis=0)
        rm_csm_sem  = np.std(rm_csm_matrix, axis=0) / np.sqrt(rm_csm_matrix.shape[0])

        time_axis_csm = np.arange(rp_csm_mean.shape[0]) / frame_rate

        #rp_csm_mean = scipy.stats.zscore(rp_csm_mean)
        
        #rm_csm_mean = scipy.stats.zscore(rm_csm_mean)




    # =========================
    if rp_csp_filtered and rm_csp_filtered:
        rp_csp_matrix = np.vstack([matrix for matrix in rp_csp_filtered.values()])
        rm_csp_matrix = np.vstack([matrix for matrix in rm_csp_filtered.values()])

        rp_csp_mean = np.mean(rp_csp_matrix, axis=0)
        rp_csp_sem  = np.std(rp_csp_matrix, axis=0) / np.sqrt(rp_csp_matrix.shape[0])

        rm_csp_mean = np.mean(rm_csp_matrix, axis=0)
        rm_csp_sem  = np.std(rm_csp_matrix, axis=0) / np.sqrt(rm_csp_matrix.shape[0])

        plotDict = {"R+": rp_csp_matrix.reshape(rp_csp_matrix.shape[0]//12,12,1000), "R-":rm_csp_matrix.reshape(rm_csp_matrix.shape[0]//12,12,1000)}

        #rp_csp_mean = scipy.stats.zscore(rp_csp_mean)         
        #rm_csp_mean = scipy.stats.zscore(rm_csp_mean)
    
    for i, cond in enumerate(["R+", "R-"]):
        print("HUHU", plotDict[cond].shape)
        nAnimal = plotDict[cond].shape[0]
        #for i in range(nAnimal):
        #    plotDict[cond][:, i, :] = scipy.stats.zscore(plotDict[cond][i, :, :])
        plotDict[cond] = np.mean(plotDict[cond], axis=0)

    minMax = []
    for i, cond in enumerate(["R+", "R-"]):
        key = cond
        xy = plotDict[key]
        #print(f"HERE: {xy.shape}")
        #xy = xy[:,1000:]
        minMax.append([xy.min(), xy.max()])
    print(minMax)
    minMax = [np.min(minMax, axis=0)[0], np.max(minMax, axis=0)[1]]
    print(minMax)

    fig, axMatrix = plt.subplots(1,2, figsize=(8,3), tight_layout=True,dpi=200)
    axList = axMatrix.flatten()

    before_tone = 10

    for i, cond in enumerate(["R+", "R-"]):
        ax = axList[i]
        key = cond
        xy = plotDict[key]
        
        imshow_plot = ax.imshow(xy, vmin=minMax[0], vmax=minMax[1], aspect="auto",interpolation='none')
        fps = 20
        ax.set_xlim(0,50*fps)
        xticks = np.arange(0,1001,1000)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks//fps-10)# minus 15 to align to tone onset
        #x.set_yticks(np.arange(4))  
        #ax.set_yticklabels([f"Ext{tick}" for tick in (np.arange(4)+1)])  
        ax.axvline(before_tone*fps, c="orange")
        ax.axvline((before_tone+10)*fps, c="orange")
        ax.axvline((before_tone+30)*fps, c="orange")
        ax.set_title(cond)
        ax.set_xlabel("Time since tone onset [sec]")
        ax.set_ylabel("Tones")
        for i in range(xy.shape[0]//12):
            ax.axhline(i*12, color="w")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.99, 0.195, 0.015, 0.68])
    fig.colorbar(imshow_plot, cax=cbar_ax, label="Neuronal activity (z-score)")
    #fig.colorbar(imshow_plot, ax=ax, location='right', anchor=(0, 0.3), label="Neuronal activity (z-score)")#, shrink=0.7)

    #fig.savefig(f"luminanceHeatmap_Rpm_sameScale_test.svg")
   

# %%
plot_rp_vs_rm_for_cs_types(
    rp_csm_stacked,
    rm_csm_stacked,
    rp_csp_stacked,
    rm_csp_stacked,
    batch_list=['A','B'],
    session="s7",
    frame_rate=20,
    verbose=True
)
# %%
