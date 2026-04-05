import tkinter as tk
from tkinter import filedialog, messagebox
import os
import ProcessingLibrary
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import numpy as np




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
    if 'cache' not in globals(): return

    # Flip the boolean
    show_corrected = not show_corrected
    
    # Update button color/text
    if show_corrected:
        btn_toggle.config(text="View: Corrected", bg="#90EE90")
    else:
        btn_toggle.config(text="View: Raw Data", bg="#FFB6C1")
    
    # RE-DRAW the current plot type (Trace, PETH, or Hist) with the new data
    universal_plot_trigger()
    
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
    
    # Button 3 is RIGHT-CLICK
    if event.button == 3: 
        is_dragging = True
        # Store the INITIAL pixel location (not data coordinates)
        press_x, press_y = event.x, event.y

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

        
def zoom_factory(ax, base_scale=1.5):
    def zoom_fun(event):
        # Only zoom if the mouse is inside the plot area
        if event.inaxes != ax: return
        
        # Get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        
        if event.button == 'up':
            # Deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # Deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1

        # Calculate new logical limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * (rel_x)])
        ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * (rel_y)])
        
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
    global ax, canvas
    
    if not ax: return
    """
    Only draws labels that are actually within the current X-axis view.
    """
    import matplotlib.transforms as transforms
    
    # Get the CURRENT visible time window
    xmin, xmax = ax.get_xlim()
    
    # Blended transform: X=Data, Y=Axis Fraction (0 to 1)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    unique_labels = set()
    
    for m in markers:
        # --- THE FIX: Skip if the marker is off-screen ---
        if m['time'] < xmin or m['time'] > xmax:
            continue
            
        legend_label = m['label'] if m['label'] not in unique_labels else "_nolegend_"
        unique_labels.add(m['label'])
        
        # 1. Vertical Line
        ax.axvline(x=m['time'], color=m['color'], linestyle='--', alpha=0.3, label=legend_label)
        
        # 2. Text Label (Pinned to top 90%)
        ax.text(m['time'], 0.9, f" {m['label']}", 
                transform=trans, 
                rotation=90, 
                verticalalignment='top', 
                fontsize=8, 
                color=m['color'], 
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

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


# --- Step 3: Plot Choice ---
plot_type_var = tk.StringVar(root)
plot_type_var.set("Continuous Trace")
plot_options = ["Continuous Trace", "PETH Heatmap", "Z-Score Distribution"]
dropdown = tk.OptionMenu(options_frame, plot_type_var, *plot_options)
dropdown.pack(side="left", padx=10)

# --- Step 4: EXECUTE ---
def universal_plot_trigger():
    choice = plot_type_var.get()
    if choice == "Continuous Trace":
        simple_plot()
    elif choice == "Z-Score Distribution":
        show_error("Histogram function not yet implemented!")

btn_plot = tk.Button(options_frame, text="2. Execute Plot", 
                     command=universal_plot_trigger, bg="#4CAF50", fg="white", font=('Helvetica', 9, 'bold'))
btn_plot.pack(side="left", padx=10)

# --- Step 5: Zoom Reset ---
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

# 3. Plotting Area
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