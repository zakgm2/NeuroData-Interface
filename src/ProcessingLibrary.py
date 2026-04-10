import tdt
import numpy as np
from scipy.optimize import curve_fit
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
    """
    High-level pipeline: Loads, extracts, debleaches, and finds markers.
    Returns a dictionary of all processed data.
    """
    # 1. Load raw structure
    data_struct = get_tdt_struct(folder_path)
    
    # 2. Extract first available stream
    store_name = list(data_struct.streams.keys())[0]
    x, y_raw, fs = get_plot_data(data_struct, store_name)
    
    # 3. Science: Apply debleaching
    y_corr, trend = correct_bleaching(y_raw, fs)
    
    # 4. Metadata: Get event markers
    markers = get_event_markers(data_struct)
    
    return {
        'x': x, 
        'raw': y_raw, 
        'corr': y_corr, 
        'trend': trend,
        'fs': fs,
        'store': store_name,
        'markers': markers
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

def get_zscore_slice(x, y, center_time, window=30):
    start_t, end_t = center_time - window, center_time + window
    
    mask = (x >= start_t) & (x <= end_t)
    slice_x = x[mask]
    slice_y = y[mask]
    
    if len(slice_y) < 2: # Need at least 2 points for std dev
        return None, None
    
    mean_val = np.mean(slice_y)
    std_val = np.std(slice_y)
    
    # Use a tiny epsilon (1e-9) to avoid true division by zero errors
    z_slice = (slice_y - mean_val) / (std_val if std_val > 0 else 1e-9)
    
    return slice_x, z_slice

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