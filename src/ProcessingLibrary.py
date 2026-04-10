import tdt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import os

def validate_tdt_folder(path):
    """
    Checks if a directory is a valid TDT data folder.
    Returns: (bool, folder_name or error_msg)
    """
    if not path:
        return False, "No folder selected."
        
    # Standard TDT folders must contain a .Tbk file (the block index)
    has_tbk = any(fname.endswith('.Tbk') for fname in os.listdir(path))
    
    if has_tbk:
        return True, os.path.basename(path)
    else:
        return False, "Invalid Folder: No TDT block files (.Tbk) found."

def process_tdt_folder(folder_path):

    data_struct = get_tdt_struct(folder_path)
    streams = data_struct.streams.keys()

    name_465 = next((s for s in streams if '465' in s), None)
    name_415 = next((s for s in streams if '415' in s), None)

    if not name_465:
        raise ValueError("No 465 signal found")

    # -------------------------
    # LOAD DATA
    # -------------------------
    x, y_465, fs = get_plot_data(data_struct, name_465)

    # -------------------------
    # 415 REGRESSION CORRECTION
    # -------------------------
    if name_415:
        _, y_415, _ = get_plot_data(data_struct, name_415)

        p = np.polyfit(y_415, y_465, 1)
        y_fit = np.polyval(p, y_415)

        # SAFE subtraction (DO NOT divide here)
        y_final = y_465 - y_fit
        display_name = f"Corrected {name_465} (via {name_415})"

    else:
        y_final = y_465
        display_name = f"{name_465} (Uncorrected)"

    # -------------------------
    # BLEACHING (trend removal)
    # -------------------------
    y_detrended, trend = correct_bleaching(y_final, fs)

    # -------------------------
    # ΔF/F NORMALIZATION (correct place)
    # -------------------------
    f0 = trend  # use fitted baseline as F0
    dff = (y_final - f0) / f0

    # -------------------------
    # REMOVE WARM-UP (fixes fuzzy start)
    # -------------------------
    start_idx = int(30 * fs)

    dff = dff[start_idx:]
    f0 = f0[start_idx:]
    x = x[start_idx:]

    # -------------------------
    # DENOISING STEP
    # -------------------------
    dff = denoise_signal(dff, fs, cutoff=5)

    # -------------------------
    # OUTPUT
    # -------------------------
    return {
        "x": x,
        "raw": y_final,
        "corr": dff,   # <-- ADD THIS LINE (compatibility)
        "dff": dff,
        "f0": f0,
        "fs": fs,
        "store": display_name,
        "markers": get_event_markers(data_struct)
        }

def double_exponential(x, a, b, c, d, k):
    """Literature-standard model for photo-bleaching decay."""
    return a * np.exp(-b * x) + c * np.exp(-d * x) + k

def get_tdt_struct(path):
    "Makes a Struct From the Binary TDT folder"
    data = tdt.read_block(path)
    if data is None:
        raise Exception("TDT returned an empty object.")
    
    #This is the raw data
    return data

def get_plot_data(data, store_name, channel=0, max_points=None):
    "Get Data From Struct"
    stream = data.streams[store_name]
    fs = stream.fs
    
    data_2d = np.atleast_2d(stream.data)
    if channel >= data_2d.shape[0]:
        channel = 0
        
    y_full = data_2d[channel, :]
    
    # Only downsample if max_points is actually provided
    if max_points:
        ds_factor = max(1, len(y_full) // max_points)
        y = y_full[::ds_factor]
        x = np.arange(len(y)) * (ds_factor / fs)
    else:
        # Get every single point for high-res analysis
        y = y_full
        x = np.arange(len(y)) / fs
    
    return x, y, fs

def correct_bleaching(y, fs):
    """
    Corrects bleaching using a masked double-exponential fit.
    Returns: (corrected_data, trend_line)
    """
    x = np.arange(len(y)) / fs
    
    # 1. MASKING: Ignore signal dips (motion/artifacts) to fit only the baseline
    threshold = np.median(y) 
    mask = y > threshold
    x_fit, y_fit = x[mask], y[mask]

    if len(y_fit) < 100:
        return y, np.zeros_like(y)

    # 2. HEURISTIC GUESSING: Makes the fit converge faster/more reliably
    k_guess = np.percentile(y_fit, 10) 
    total_amp = np.max(y_fit) - k_guess
    p0 = [total_amp*0.6, 0.05, total_amp*0.4, 0.0001, k_guess]
    
    # 3. CONSTRAINED OPTIMIZATION: Prevents the curve from 'diving' to zero
    lower = [0, 0, 0, 0, k_guess * 0.8]
    upper = [np.inf, 1, np.inf, 0.1, np.max(y_fit)]

    try:
        popt, _ = curve_fit(double_exponential, x_fit, y_fit, p0=p0, 
                            bounds=(lower, upper), maxfev=10000)
        trend = double_exponential(x, *popt)
    except Exception:
        # Fallback to single exponential via log-linear fit if optimization fails
        coeffs = np.polyfit(x_fit, np.log(np.maximum(y_fit, 1e-6)), 1)
        trend = np.exp(np.polyval(coeffs, x))

    # 4. NORMALIZATION: Centers the data around the mean of the fit-segment
    corrected = y - trend + trend[0]
    return corrected, trend

def get_event_markers(data):
    """
    Returns a list of dictionaries for every note found in the TDT file.
    """
    if not hasattr(data.epocs, 'Note'):
        return []
    
    notes = data.epocs.Note.notes
    onsets = data.epocs.Note.onset
    
    markers = []
    # Simple color map for different notes
    color_map = {'Clap': 'red', 'Sucrose': 'green', 'Stop': 'blue'}
    
    for n, t in zip(notes, onsets):
        note_str = n.decode() if isinstance(n, bytes) else str(n)
        note_str = note_str.strip()
        
        markers.append({
            'time': t,
            'label': note_str,
            'color': color_map.get(note_str, 'black') # Default to black if unknown
        })
    return markers

def get_zscore_slice(time_array, signal, center_t, window=30):
    start_idx = np.searchsorted(time_array, center_t - window)
    end_idx = np.searchsorted(time_array, center_t + window)
    
    seg_y = signal[start_idx:end_idx]
    seg_x = time_array[start_idx:end_idx]

    # 1. THE ARTIFACT CLIPPER
    # Brains rarely go above 15-20 Z-scores. 
    # If the raw dF/F is over 100%, it's almost certainly a cable bump.
    seg_y = np.clip(seg_y, -1.0, 1.0) # Limits raw dF/F to 100% change

    # 2. Baseline Calculation
    baseline_split = len(seg_y) // 2
    baseline_period = seg_y[:baseline_split]
    
    mu = np.mean(baseline_period)
    std = np.std(baseline_period)
    
    # 3. Handle the "Absolute Mess" (High Noise)
    # If the baseline is too noisy (std is huge), we can flag it
    if std > 0.5: # Adjust this threshold based on your typical noise
        print(f"Warning: High noise detected at {center_t}s. Result may be messy.")

    if std == 0: return seg_x, np.zeros_like(seg_y)
    
    z_scored_seg = (seg_y - mu) / std
    return seg_x, z_scored_seg

def smooth_signal(data, fs, window_sec=0.5):
    """Calculates a moving average smoothed signal."""
    window_size = int(fs * window_sec)
    if window_size % 2 == 0: window_size += 1
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def bin_for_heatmap(z_seg, num_bins=300):
    """Calculates the averages for the heatmap bins."""
    if z_seg is None or len(z_seg) == 0: return np.zeros(num_bins)
    bin_edges = np.linspace(0, len(z_seg), num_bins + 1).astype(int)
    return np.array([np.mean(z_seg[bin_edges[i]:bin_edges[i+1]]) for i in range(num_bins)])

def denoise_signal(signal, fs, cutoff=5, order=2):
    """
    Light low-pass filter for photometry ΔF/F signals.

    cutoff: Hz (5 Hz is standard safe range for behavior tasks)
    """

    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)

    return filtered
