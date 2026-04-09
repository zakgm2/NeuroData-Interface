import tkinter as tk
from tkinter import filedialog, messagebox
import os
import ProcessingLibrary
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import numpy as np
import datetime



#####################        
# ---  Globals  --- #
#####################

show_corrected = True
folder_path = None
current_data = None
is_dragging = False
press_x, press_y = None, None
  
root = tk.Tk()
root.title("NeuroData Interface - " + str(folder_path))

def toggle_bleaching_action():
    global show_corrected
    if 'cache' not in globals() or cache is None: return
    
    # 1. ONLY capture the Time (X) view
    curr_xlim = ax.get_xlim()
    
    # 2. Toggle the data
    show_corrected = not show_corrected
    
    # 3. Re-plot
    simple_plot() 
    
    # 4. Restore the Time zoom, but let Y-axis auto-scale to the new data height
    ax.set_xlim(curr_xlim)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    
    canvas.draw_idle()
    
def on_select(eclick, erelease):
    """
    Called when you finish dragging the rectangle.
    """
    # Get the coordinates in DATA units
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    # If the user clicks outside the actual plot area, xdata/ydata will be None
    if None in [x1, x2, y1, y2]:
        return

    # Prevent 'micro-zooms' from accidental clicks
    if abs(x1 - x2) < 0.1:
        return

    # Apply the new window
    ax.set_xlim(min(x1, x2), max(x1, x2))
    ax.set_ylim(min(y1, y2), max(y1, y2))
    
    canvas.draw_idle()

def on_press(event):
    global is_dragging, press_x, press_y
    if event.inaxes != ax: return
    
    # 1. Start dragging logic for Right-Click (Button 3)
    if event.button == 3: 
        is_dragging = True
        press_x, press_y = event.x, event.y

    # 2. Check for double-click separately 
    # We do NOT return early here so the rest of the script stays alive
    if event.dblclick and event.button == 1:
        launch_zscore_peth(event.xdata)
        
def on_motion(event):
    # 1. ADD CACHE HERE so the function can see your data
    global ax, canvas, is_dragging, press_x, press_y, cache
    
    if not is_dragging or event.inaxes != ax: 
        return
    if event.x is None or event.y is None: 
        return

    # --- (Keep your existing dx/dy and shift_x/y math here) ---
    dx = event.x - press_x
    dy = event.y - press_y
    press_x, press_y = event.x, event.y

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    bbox = ax.get_window_extent()
    
    shift_x = (dx / bbox.width) * (cur_xlim[1] - cur_xlim[0])
    shift_y = (dy / bbox.height) * (cur_ylim[1] - cur_ylim[0])

    ax.set_xlim(cur_xlim[0] - shift_x, cur_xlim[1] - shift_x)
    ax.set_ylim(cur_ylim[0] - shift_y, cur_ylim[1] - shift_y)

    # 2. THE REFRESH LOGIC:
    # We clear the old "Clap" lines/text so they don't stack up and lag
    for artist in ax.lines + ax.texts:
        if isinstance(artist, plt.Line2D) and artist.get_linestyle() == '--':
            artist.remove()
    while ax.texts:
        ax.texts[0].remove()
        
    # 3. REDRAW ONLY IF CACHE EXISTS:
    # This brings the 'Clap' and 'Food' labels back as you move
    if 'cache' in globals() and cache is not None:
        _update_plot_with_notes(cache['markers'])
    
    canvas.draw_idle()
    
def on_release(event):
    global is_dragging
    is_dragging = False

def launch_zscore_peth(center_t):
    if 'cache' not in globals() or cache is None: return
    
    # --- 1. SMART DATA SELECTION ---
    # This ensures the PETH matches exactly what you are seeing on the main screen
    is_corr = show_corrected  # Checks your global toggle
    data_source = cache['corr'] if is_corr else cache['raw']
    fs = cache['fs']
    x_full = cache['x']
    
    # --- 2. SMOOTHING & WINDOWING ---
    window_size = int(fs * 0.5) 
    if window_size % 2 == 0: window_size += 1
    clean_signal = np.convolve(data_source, np.ones(window_size)/window_size, mode='same')
    
    start_t, end_t = center_t - 30, center_t + 30
    mask = (x_full >= start_t) & (x_full <= end_t)
    y_seg = clean_signal[mask]
    
    if len(y_seg) > 10:
        # --- 3. ROBUST Z-SCORE ---
        median_val = np.median(y_seg)
        mad_val = np.median(np.abs(y_seg - median_val))
        z_seg = 0.6745 * (y_seg - median_val) / (mad_val if mad_val != 0 else 1)
        z_seg = np.clip(z_seg, -5, 5)

        # --- 4. BINNING ---
        num_bins = 300 
        bin_edges = np.linspace(0, len(z_seg), num_bins + 1).astype(int)
        z_binned = np.array([np.mean(z_seg[bin_edges[i]:bin_edges[i+1]]) for i in range(num_bins)])

        # --- 5. UI SETUP ---
        mode_str = "Corrected" if is_corr else "Raw"
        pop = tk.Toplevel(root)
        pop.title(f"Z-score Peth ({mode_str}) - {center_t:.2f}s")
        
        fig_peth = Figure(figsize=(8, 7), dpi=100)
        ax_heat, ax_line = fig_peth.subplots(2, 1, sharex=True, 
                                           gridspec_kw={'height_ratios': [1, 1]})
        
        # Plotting Heatmap
        ax_heat.imshow(z_binned.reshape(1, -1), aspect='auto', 
                       cmap='YlGnBu_r', extent=[-30, 30, 0, 1],
                       vmin=-5, vmax=5, interpolation='bilinear') 
        
        ticks = [-30, -20, -10, 0, 10, 20, 30]
        ax_heat.set_xticks(ticks)
        ax_heat.set_xticklabels(ticks)
        ax_heat.set_xlabel("Trial #", fontweight='bold')
        ax_heat.set_yticks([]) 
        
        # Plotting Line
        x_axis = np.linspace(-30, 30, len(z_seg))
        ax_line.plot(x_axis, z_seg, color='black', linewidth=2) 
        ax_line.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        
        ax_line.set_ylim([-5, 5])
        ax_line.set_ylabel(f"Z-Score ({mode_str})", fontweight='bold')
        ax_line.set_xlabel("Time from Center (s)", fontweight='bold')
        ax_line.grid(True, linestyle=':', alpha=0.6)
        
        fig_peth.suptitle(f"Z-score Peth ({mode_str} Data)", fontsize=14, fontweight='bold')
        fig_peth.tight_layout(rect=[0, 0.05, 1, 0.95]) 
        
        canvas_peth = FigureCanvasTkAgg(fig_peth, master=pop)
        canvas_peth.draw()
        canvas_peth.get_tk_widget().pack(fill="both", expand=True)

        # --- 6. EXPORT LOGIC ---
        def save_peth_action():
            import datetime
            import os
            
            experiment_folder = os.path.dirname(folder_path) if folder_path else os.getcwd()
            ts = datetime.datetime.now().strftime("%H%M%S")
            
            # Filename now includes if it was Raw or Corrected
            default_fn = f"PETH_{mode_str}_{int(center_t)}s_{ts}.png"
            
            fpath = filedialog.asksaveasfilename(
                initialdir=experiment_folder,
                defaultextension=".png",
                initialfile=default_fn,
                title=f"Export {mode_str} PETH Analysis"
            )
            
            if fpath:
                fig_peth.savefig(fpath, dpi=300, bbox_inches='tight')

        # Button Frame
        btn_frame = tk.Frame(pop)
        btn_frame.pack(side="bottom", fill="x", pady=10)
        
        btn_export_peth = tk.Button(
            btn_frame, text=f"💾 Export {mode_str} PETH", 
            command=save_peth_action, bg="#2196F3", fg="white", 
            font=('Helvetica', 10, 'bold'), padx=20
        )
        btn_export_peth.pack()
def zoom_factory(ax, base_scale=1.2):
    def zoom_fun(event):
        global canvas
        # event.x and event.y are pixel locations
        # bbox is the plot's physical box on screen
        bbox = ax.get_window_extent()
        
        # 1. Determine the "Zone" based on pixels
        is_on_x_axis = event.y < bbox.ymin  # Below the plot
        is_on_y_axis = event.x < bbox.xmin  # Left of the plot
        is_inside = event.inaxes == ax      # Inside the plot

        # 2. Set scale factor
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        # --- MODE A: STRETCH X (Scroll below the graph) ---
        if is_on_x_axis and not is_on_y_axis:
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            # We zoom toward the mouse's X-position in data units
            xdata = event.xdata if event.xdata else (cur_xlim[0] + cur_xlim[1]) / 2
            rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])

        # --- MODE B: STRETCH Y (Scroll left of the graph) ---
        elif is_on_y_axis and not is_on_x_axis:
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            ydata = event.ydata if event.ydata else (cur_ylim[0] + cur_ylim[1]) / 2
            rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])

        # --- MODE C: UNIFORM ZOOM (Inside the graph) ---
        elif is_inside:
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            rel_x = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
            rel_y = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([event.xdata - new_width * (1 - rel_x), event.xdata + new_width * rel_x])
            ax.set_ylim([event.ydata - new_height * (1 - rel_y), event.ydata + new_height * rel_y])

        canvas.draw_idle()

    return zoom_fun

# FOR ERRORS (Red X icon)
def show_error(msg):
    tk.messagebox.showerror("Data Error", msg)

# FOR SUCCESS (Blue 'i' icon)
def show_success(msg):
    tk.messagebox.showinfo("Success", msg)

# CHOOSING FILE TO USE
def choose_file():
    global folder_path
    path = filedialog.askdirectory()
    
    # Validation: Check for TDT files
    if path and any(fname.endswith('.Tbk') for fname in os.listdir(path)):
        folder_path = path
        root.title("NeuroData Interface - " + os.path.basename(folder_path))
        show_success(f"Successfully linked to: {os.path.basename(path)}")
    else:
        # This triggers the classic Windows 'Ding' and Error Window
        show_error("Invalid Folder: No TDT data files (.Tbk) found in this directory.")

#LOADING THE DATA INTO THE GUI
def load_data_action():
    global current_data, cache
    if folder_path is None:
        show_error("Please select a folder first!")
        return

    try:
        # 1. Load the raw TDT structure
        current_data = ProcessingLibrary.get_tdt_struct(folder_path)
        
        # 2. Automatically pick the first stream (e.g., '465N' or 'Fi1r')
        store_name = list(current_data.streams.keys())[0]
        
        # 3. Get the raw Y, the time X, and the FS
        x, y_raw, fs = ProcessingLibrary.get_plot_data(current_data, store_name)
        
        # 4. Apply the 'Real' Bleaching Correction we built
        # This uses the mask to ignore those 0-dips
        y_corr, trend = ProcessingLibrary.correct_bleaching(y_raw, fs)

        # 5. Get the notes
        event_markers = ProcessingLibrary.get_event_markers(current_data)
        
        # 6. Store in cache for the Plotting function
        cache = {
            'x': x, 
            'raw': y_raw, 
            'corr': y_corr, 
            'trend': trend,
            'fs': fs,
            'store': store_name,
            'markers': event_markers
        }
        
        show_success(f"Successfully loaded {store_name} ({fs:.2f} Hz)")
        
    except Exception as e:
        show_error(f"Load Error: {str(e)}")


#LOADING THE DATA IN THE GUI
def load_data_set():
    choose_file()
    load_data_action()
    return

#ADDS NOTES TO THE PLOT
def _update_plot_with_notes(markers):
    global ax
    import matplotlib.transforms as transforms
    
    # 1. Get the CURRENT visible time window
    xmin, xmax = ax.get_xlim()
    
    # 2. Setup the "Sticky" transform (X=Data, Y=Axis Fraction)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    unique_labels = set()
    
    for m in markers:
        # --- FIX 1: TIME FILTER ---
        # If the clap is more than 0.5s outside the current view, skip it
        if m['time'] < xmin or m['time'] > xmax:
            continue
            
        legend_label = m['label'] if m['label'] not in unique_labels else "_nolegend_"
        unique_labels.add(m['label'])
        
        # 3. Draw the Line
        ax.axvline(x=m['time'], color=m['color'], linestyle='--', alpha=0.3, label=legend_label)
        
        # 4. --- FIX 2: THE CLIP BOX ---
        # 'clip_on=True' tells Matplotlib to chop off the text if it hits the axis edge
        ax.text(m['time'], 0.95, f" {m['label']}", 
                transform=trans, 
                rotation=90, 
                verticalalignment='top', 
                clip_on=True,              # <--- Crucial for zooming
                fontsize=8, 
                color=m['color'], 
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

def _refresh_notes_position():
    """
    Adjusts existing text labels to stay at the top of the current zoom level.
    Called during dragging (on_motion).
    """
    if not ax.texts: return
    
    ymin, ymax = ax.get_ylim()
    new_y = ymin + (ymax - ymin) * 0.9  # Keep at 90% height
    
    for txt in ax.texts:
        txt.set_y(new_y)

#PLOT CORRECTED FOR BLEACHING        
def simple_plot():
    if 'cache' not in globals(): return
    
    ax.clear()
    
    # Decide which data to pull from the cache
    data_to_plot = cache['corr'] if show_corrected else cache['raw']
    label_text = "Corrected (Debleached)" if show_corrected else "Raw Signal"
    color_choice = "blue" if show_corrected else "gray"

    # Horizontal line at 0 (Amplitude)
    ax.axhline(0, color='black', linewidth=1.2, alpha=0.5, zorder=1)
    # Vertical line at 0 (Time start)
    ax.axvline(0, color='black', linewidth=1.2, alpha=0.5, zorder=1)
    
    ax.plot(cache['x'], data_to_plot, color=color_choice, alpha=0.8, label=label_text)
    
    # Still show notes on both!
    _update_plot_with_notes(cache['markers'])
    
    ax.set_title(f"{label_text} - {cache['store']}")
    canvas.draw()

def export_canvas_action():
    global fig, folder_path, cache
    
    if 'cache' not in globals() or cache is None:
        show_error("No data loaded to export!")
        return

    # 1. Create a timestamp (YearMonthDay_HourMinute)
    # Example: 20260408_2209 for April 8th at 10:09 PM
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # 2. Build the default filename
    initial_name = f"{cache['store']}_Plot_{timestamp}.png"
    
    # 3. Open the Save Dialog
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("SVG Vector", "*.svg")],
        initialfile=initial_name,
        title="Export Current View"
    )

    if file_path:
        try:
            # Save with high resolution (300 DPI) for your thesis/proposal
            fig.savefig(file_path, dpi=300, bbox_inches='tight', transparent=False)
            show_success(f"Successfully exported:\n{os.path.basename(file_path)}")
        except Exception as e:
            show_error(f"Export Failed: {str(e)}")
#####################        
# --- GUI SETUP --- #
#####################

# 1. Window Configuration
window_width = 1250
window_height = 850 # Made slightly taller for options
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

# 2. Control Panel (Options Frame)
options_frame = tk.LabelFrame(root, text="Controls & Analysis")
options_frame.pack(pady=10, padx=10, fill="x")

# --- Step 1: Loading ---
btn_choose = tk.Button(options_frame, text="1. Select & Load TDT Folder", 
                       command=load_data_set, bg="#e1e1e1")
btn_choose.pack(side="left", padx=10, pady=10)

tk.Label(options_frame, text="|").pack(side="left", padx=5)


# ---  Plot Choice ---
plot_type_var = tk.StringVar(root)
plot_type_var.set("Continuous Trace")
plot_options = ["Continuous Trace", "PETH Heatmap", "Z-Score Distribution"]
dropdown = tk.OptionMenu(options_frame, plot_type_var, *plot_options)
dropdown.pack(side="left", padx=10)

# --- EXECUTE ---
def universal_plot_trigger():
    choice = plot_type_var.get()
    if choice == "Continuous Trace":
        simple_plot()
    elif choice == "Z-Score Distribution":
        show_error("Histogram function not yet implemented!")

btn_plot = tk.Button(options_frame, text="2. Execute Plot", 
                     command=universal_plot_trigger, bg="#4CAF50", fg="white", font=('Helvetica', 9, 'bold'))
btn_plot.pack(side="left", padx=10)

# ---  Zoom Reset ---
def reset_zoom():
    if 'cache' not in globals(): return

    # 1. Clear the graph
    ax.clear()
    
    # 2. Re-run the current plot (Trace, PETH, or Hist)
    # This automatically resets the axes to 'Full Scale'
    universal_plot_trigger()
    
    # 3. Re-link the rectangle selector so it doesn't break
    if 'rect_selector' in globals():
        rect_selector.ax = ax
        rect_selector.set_active(True)

    canvas.draw_idle()
        
btn_toggle = tk.Button(options_frame, text="Toggle Bleaching", 
                       command=toggle_bleaching_action, bg="#90EE90")
btn_toggle.pack(side="left", padx=10)

btn_reset = tk.Button(options_frame, text="Reset Zoom", command=reset_zoom)
btn_reset.pack(side="left", padx=10)


btn_export = tk.Button(
    options_frame, 
    text="Export View (PNG/PDF)", 
    command=export_canvas_action, 
    bg="#2196F3",   # A nice professional blue
    fg="white", 
    font=('Helvetica', 9, 'bold')
)
btn_export.pack(side="left", padx=10)


#     Plotting Area
plot_frame = tk.Frame(root)
plot_frame.pack(side="bottom", fill="both", expand=True, pady=10)

fig = Figure(figsize=(8, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)



# --- Initialize the  Zoomings ---
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# Connect Scrolling
f_zoom = zoom_factory(ax, base_scale=1.2)
fig.canvas.mpl_connect('scroll_event', f_zoom)

# Rectangle Selector for Left-Click Zoom
global rect_selector
rect_selector = RectangleSelector(
    ax, 
    on_select, 
    useblit=True,
    button=[1],           # 1 = Left-Click only! No more snap-back.
    minspanx=5, 
    minspany=0.001,
    # spandata=True,      <-- REMOVED: This causes the crash in new Matplotlib
    props=dict(facecolor='yellow', edgecolor='black', alpha=0.3, fill=True),
    interactive=True
)
rect_selector.set_active(True)



root.mainloop()