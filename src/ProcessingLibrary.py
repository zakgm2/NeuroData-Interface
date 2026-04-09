import tdt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy as scp
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt



def double_exponential(x, a, b, c, d, k):
    "The physical model of fluorophore bleaching."
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
    "Corrects Bleaching according to literature-stated decay"
    x = np.arange(len(y)) / fs
    
    #1. THE RIGID MASK: Only look at the "Top" 50% of the signal
    # This ensures the 0s are invisible to the math
    threshold = np.median(y) 
    mask = y > threshold
    
    x_fit = x[mask]
    y_fit = y[mask]

    if len(y_fit) < 100:
        return y, np.zeros_like(y)

    #2. SMART INITIAL GUESS: Instead of fixed numbers, we look at the data
    #k (baseline) should be near the end of the recording
    k_guess = np.percentile(y_fit, 10) 
    
    #Total Amplitude to be explained by the decay
    total_amp = np.max(y_fit) - k_guess
    
    #p0 = [Amp_fast, Decay_fast, Amp_slow, Decay_slow, Baseline]
    p0 = [total_amp*0.6, 0.05, total_amp*0.4, 0.0001, k_guess]
    
    #3. BOUNDS: Prevent the "Drop to Zero"
    # We force the baseline 'k' to be at least the bottom of our HIGH signal
    lower_bounds = [0, 0, 0, 0, k_guess * 0.8]
    upper_bounds = [np.inf, 1, np.inf, 0.1, np.max(y_fit)]

    try:
        popt, _ = curve_fit(double_exponential, x_fit, y_fit, p0=p0, 
                            bounds=(lower_bounds, upper_bounds), maxfev=10000)
        trend = double_exponential(x, *popt)
    except:
        # If double-exp fails, fit a single exponential (simpler/more stable)
        coeffs = np.polyfit(x_fit, np.log(np.maximum(y_fit, 1e-6)), 1)
        trend = np.exp(np.polyval(coeffs, x))

    #4. RESTORE: Keep the signal at the "High" mean
    corrected = y - trend + np.mean(y_fit)
    
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
    """
    Slices a signal y around a center_time (+/- window) 
    and returns a local z-score.
    """
    # Calculate start and end indices based on the time vector x
    # This is more robust than just index math if your sampling isn't perfectly uniform
    start_time = center_time - window
    end_time = center_time + window
    
    mask = (x >= start_time) & (x <= end_time)
    slice_y = y[mask]
    
    if len(slice_y) == 0:
        return None
    
    # Calculate local Z-score
    mean_val = np.mean(slice_y)
    std_val = np.std(slice_y)
    
    if std_val == 0: 
        return np.zeros_like(slice_y)
        
    z_slice = (slice_y - mean_val) / std_val
    return z_slice
