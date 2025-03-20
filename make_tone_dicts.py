import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# Session Extraction: extract s4, s5, s6, s7 from the sessions dict
# -------------------------------------------------------------------

def extract_sessions_subset(sessions_dict, session_ids, deepcopy=True):
    """
    Extracts a subset of sessions (s4, s5, s6, s7) from the given sessions_dict.

    Parameters:
    - sessions_dict (dict): Original dictionary with all session data.
    - session_ids (list): List of session identifiers (e.g., ['s4', 's5', 's6', 's7']).
    - deepcopy (bool): Whether to deep copy the extracted sessions.

    Returns:
    - ext_sessions_dict (dict): Extracted session traces.
    """
    ext_sessions_dict = {}

    for key in sessions_dict.keys():
        key_parts = key.split('_')

        if len(key_parts) >= 3 and key_parts[2] in session_ids:
            if deepcopy:
                ext_sessions_dict[key] = copy.deepcopy(sessions_dict[key])
            else:
                ext_sessions_dict[key] = sessions_dict[key]

    print(f"-Extracted {len(ext_sessions_dict)} sessions for {session_ids}")
    return ext_sessions_dict

# -------------------------------------------------------------------
# Tone Extraction: CS- and CS+ traces with optional surrounding
# -------------------------------------------------------------------

def extract_tones_with_surrounding(ext_sessions_dict, 
                                    surrounding_before=0, 
                                    surrounding_after=0, 
                                    return_as_dataframe=False):
    """
    Extracts CS- and CS+ tone-playing parts along with optional surrounding frames 
    for each neuron trace.

    Parameters:
    - ext_sessions_dict (dict): Dictionary with only the selected session data (s4-s7).
    - surrounding_before (int): Number of frames to include before each tone (default: 0).
    - surrounding_after (int): Number of frames to include after each tone (default: 0).
    - return_as_dataframe (bool): Return result as pandas DataFrame with neuron labels.

    Returns:
    - csm_sessions_dict_with_surrounding (dict): Extracted CS- traces + optional surrounding.
    - csp_sessions_dict_with_surrounding (dict): Extracted CS+ traces + optional surrounding.
    """
    
    # Define key frame indices (0-based)
    csm_start = 3599   # CS- starts at frame 3600 (0-indexed)
    csp_start = 10_800 # CS+ starts after CS-

    min_frames_required = 32_382 # Safety check for minimum frame length

    # Tone protocol config
    csm_tones = 4
    csp_tones = 12
    tone_duration = 600    # Frames per tone
    pause_duration = 1200  # Frames per pause

    csm_sessions_dict_with_surrounding = {}
    csp_sessions_dict_with_surrounding = {}

    for key, traces in ext_sessions_dict.items():
        if isinstance(traces, pd.DataFrame):
            traces_np = traces.to_numpy()
            neuron_labels = traces.columns
        else:
            traces_np = traces
            neuron_labels = None

        num_frames, num_neurons = traces_np.shape

        if num_frames < min_frames_required:
            print(f"!!WARNING: Session '{key}' has only {num_frames} frames (expected at least {min_frames_required})")

        mask_csm = np.zeros(num_frames, dtype=bool)
        mask_csp = np.zeros(num_frames, dtype=bool)

        # CS- tones extraction
        for i in range(csm_tones):
            start = max(0, csm_start + i * (tone_duration + pause_duration) - surrounding_before)
            end = min(num_frames, csm_start + i * (tone_duration + pause_duration) + tone_duration + surrounding_after)
            mask_csm[start:end] = True

        # CS+ tones extraction
        for i in range(csp_tones):
            start = max(0, csp_start + i * (tone_duration + pause_duration) - surrounding_before)
            end = min(num_frames, csp_start + i * (tone_duration + pause_duration) + tone_duration + surrounding_after)
            mask_csp[start:end] = True

        # Extract the traces based on masks
        csm_traces = traces_np[mask_csm, :]
        csp_traces = traces_np[mask_csp, :]

        if return_as_dataframe and neuron_labels is not None:
            csm_traces = pd.DataFrame(csm_traces, columns=neuron_labels)
            csp_traces = pd.DataFrame(csp_traces, columns=neuron_labels)

        csm_sessions_dict_with_surrounding[key] = csm_traces
        csp_sessions_dict_with_surrounding[key] = csp_traces

        print(f"✅ {key}: CS- shape {csm_traces.shape}, CS+ shape {csp_traces.shape}")

    return csm_sessions_dict_with_surrounding, csp_sessions_dict_with_surrounding


""" with rp and rm grouping:"""
def extract_tones_with_surrounding_and_grouping(
    ext_sessions_dict,
    surrounding_before=0,
    surrounding_after=0,
    return_as_dataframe=False,
    verbose=True
):
    """
    Extracts CS- and CS+ tone parts with optional surrounding frames,
    and splits the data into rp and rm sessions.

    Returns:
    - rp_csm_sessions_dict
    - rm_csm_sessions_dict
    - rp_csp_sessions_dict
    - rm_csp_sessions_dict

    Extracts CS- and CS+ tone parts with optional surrounding frames,
    and splits the data into rp and rm sessions.

    Parameters:
    - ext_sessions_dict (dict): Dictionary with session traces.
    - surrounding_before (int): Frames before tone to include.
    - surrounding_after (int): Frames after tone to include.
    - return_as_dataframe (bool): Return data as pandas DataFrame if True.
    - verbose (bool): Print details for each session.

    Returns:
    - rp_csm_sessions_dict (dict): CS- traces for rp sessions.
    - rm_csm_sessions_dict (dict): CS- traces for rm sessions.
    - rp_csp_sessions_dict (dict): CS+ traces for rp sessions.
    - rm_csp_sessions_dict (dict): CS+ traces for rm sessions.



    """

    # Init grouped dictionaries
    rp_csm_sessions_dict = {}
    rm_csm_sessions_dict = {}
    rp_csp_sessions_dict = {}
    rm_csp_sessions_dict = {}

    # Define key frame indices (0-based)
    csm_start = 3599
    csp_start = 10_800
    min_frames_required = 32_382

    # Tone protocol config
    csm_tones = 4
    csp_tones = 12
    tone_duration = 600
    pause_duration = 1200

    for key, traces in ext_sessions_dict.items():
        if isinstance(traces, pd.DataFrame):
            traces_np = traces.to_numpy()
            neuron_labels = traces.columns
        else:
            traces_np = traces
            neuron_labels = None

        num_frames, num_neurons = traces_np.shape

        if num_frames < min_frames_required:
            print(f"!!WARNING: Session '{key}' has only {num_frames} frames (expected at least {min_frames_required})")

        mask_csm = np.zeros(num_frames, dtype=bool)
        mask_csp = np.zeros(num_frames, dtype=bool)

        # CS- tones
        for i in range(csm_tones):
            start = max(0, csm_start + i * (tone_duration + pause_duration) - surrounding_before)
            end = min(num_frames, csm_start + i * (tone_duration + pause_duration) + tone_duration + surrounding_after)
            mask_csm[start:end] = True

        # CS+ tones
        for i in range(csp_tones):
            start = max(0, csp_start + i * (tone_duration + pause_duration) - surrounding_before)
            end = min(num_frames, csp_start + i * (tone_duration + pause_duration) + tone_duration + surrounding_after)
            mask_csp[start:end] = True

        csm_traces = traces_np[mask_csm, :]
        csp_traces = traces_np[mask_csp, :]

        if return_as_dataframe and neuron_labels is not None:
            csm_traces = pd.DataFrame(csm_traces, columns=neuron_labels)
            csp_traces = pd.DataFrame(csp_traces, columns=neuron_labels)

        # Grouping based on 'rp' or 'rm' in key
        if '_rp' in key:
            rp_csm_sessions_dict[key] = csm_traces
            rp_csp_sessions_dict[key] = csp_traces
        elif '_rm' in key:
            rm_csm_sessions_dict[key] = csm_traces
            rm_csp_sessions_dict[key] = csp_traces
        else:
            print(f"!!WARNING: Could not determine group for {key}")

        if verbose:
            print(f"OK {key}: CS- shape {csm_traces.shape}, CS+ shape {csp_traces.shape}")

    return rp_csm_sessions_dict, rm_csm_sessions_dict, rp_csp_sessions_dict, rm_csp_sessions_dict

# -------------------------------------------------------------------
# Optional: Heatmap Generation for Visualizing the Tones
# -------------------------------------------------------------------

def generate_heatmap_csm(csm_sessions_dict, title: str, frame_rate=20, tone_surrounding=200):
    """
    Generates heatmaps for CS- tone traces for R+ and R- sessions.
    """
    r_plus_keys = [k for k in csm_sessions_dict if k.endswith('_rp')]
    r_minus_keys = [k for k in csm_sessions_dict if k.endswith('_rm')]

    if not r_plus_keys or not r_minus_keys:
        print(" !No R+ or R- sessions found!")
        return

    r_plus_traces = np.vstack([csm_sessions_dict[k] for k in r_plus_keys])
    r_minus_traces = np.vstack([csm_sessions_dict[k] for k in r_minus_keys])

    num_frames = r_plus_traces.shape[1]
    time_axis = np.arange(num_frames) / frame_rate

    tone_onsets = np.array([tone_surrounding + i * (600 + 1200) for i in range(4)]) / frame_rate

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    sns.heatmap(r_plus_traces, ax=axes[0], cmap="coolwarm", cbar=True)
    axes[0].set_title("Resilient (R+) - CS- Tones")
    for t in tone_onsets:
        axes[0].axvline(x=t * frame_rate, color='red', linestyle='--')

    sns.heatmap(r_minus_traces, ax=axes[1], cmap="coolwarm", cbar=True)
    axes[1].set_title("Non-Resilient (R-) - CS- Tones")
    for t in tone_onsets:
        axes[1].axvline(x=t * frame_rate, color='red', linestyle='--')

    axes[1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()

def generate_heatmap_csp(csp_sessions_dict, title: str, frame_rate=20, tone_surrounding=200):
    """
    Generates heatmaps for CS+ tone traces for R+ and R- sessions.
    """
    r_plus_keys = [k for k in csp_sessions_dict if k.endswith('_rp')]
    r_minus_keys = [k for k in csp_sessions_dict if k.endswith('_rm')]

    if not r_plus_keys or not r_minus_keys:
        print("⚠! No R+ or R- sessions found!")
        return

    r_plus_traces = np.vstack([csp_sessions_dict[k] for k in r_plus_keys])
    r_minus_traces = np.vstack([csp_sessions_dict[k] for k in r_minus_keys])

    num_frames = r_plus_traces.shape[1]
    time_axis = np.arange(num_frames) / frame_rate

    tone_onsets = np.array([tone_surrounding + i * (600 + 1200) for i in range(12)]) / frame_rate

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    sns.heatmap(r_plus_traces, ax=axes[0], cmap="coolwarm", cbar=True)
    axes[0].set_title("Resilient (R+) - CS+ Tones")
    for t in tone_onsets:
        axes[0].axvline(x=t * frame_rate, color='red', linestyle='--')

    sns.heatmap(r_minus_traces, ax=axes[1], cmap="coolwarm", cbar=True)
    axes[1].set_title("Non-Resilient (R-) - CS+ Tones")
    for t in tone_onsets:
        axes[1].axvline(x=t * frame_rate, color='red', linestyle='--')

    axes[1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()




