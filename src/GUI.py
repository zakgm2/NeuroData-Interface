import tkinter as tk
from tkinter import filedialog
import os
import ProcessingLibrary
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
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

def show_window_toast(message, duration=2500):
    """
    Displays a modern, non-blocking notification in the bottom-right 
    of the main window. Replaces clunky 'OK' popups.
    """
    # 1. Create a borderless window
    toast = tk.Toplevel(root)
    toast.overrideredirect(True) # Removes the title bar/minimize buttons
    toast.attributes("-topmost", True) # Keeps it above the graph
    
    # 2. Style it (Dark theme looks very professional)
    label = tk.Label(toast, text=message, bg="#333333", fg="white", 
                     padx=20, pady=10, font=("Helvetica", 10, "bold"))
    label.pack()

    # 3. POSITIONING MATH
    # Force Tkinter to calculate the sizes before we use them
    root.update_idletasks()
    toast.update_idletasks()
    
    # Get coordinates of the main window
    root_x = root.winfo_x()
    root_y = root.winfo_y()
    root_w = root.winfo_width()
    root_h = root.winfo_height()
    
    # Get size of the toast itself
    t_w = toast.winfo_width()
    t_h = toast.winfo_height()
    
    # Calculate bottom-right with 20px padding
    pos_x = root_x + root_w - t_w - 20
    pos_y = root_y + root_h - t_h - 20
    
    toast.geometry(f"+{pos_x}+{pos_y}")
    
    # 4. Auto-destroy after the duration (ms)
    toast.after(duration, toast.destroy)

def toggle_bleaching_action():
    global show_corrected
    if 'cache' not in globals() or cache is None: return
    
    # 1. THE MEMORY: Grab exactly where the user is looking
    curr_xlim = ax.get_xlim()
    curr_ylim = ax.get_ylim()
    
    # 2. THE SWAP: Flip the boolean
    show_corrected = not show_corrected
    
    # 3. THE RE-RENDER: Call your plot function
    simple_plot(draw_now=False)
    
    # 4. THE RESTORATION: Force the "camera" back to the previous zoom
    ax.set_xlim(curr_xlim)
    ax.set_ylim(curr_ylim)
    
    canvas.draw_idle()
    
    # 6. Toast Feedback (Replaces intrusive popups)
    mode = "Debleached" if show_corrected else "Raw"
    show_window_toast(f"View Switched: {mode}")
    
def on_select(eclick, erelease):
    """
    Handles the 'Box Zoom' logic when the user drags a rectangle.
    """
    # 1. Coordinate Validation
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    if None in [x1, x2, y1, y2]:
        return

    # 2. Prevent 'Micro-Zooms' (Accidental clicks while trying to drag)
    # 0.1s is a good threshold for photometry data
    if abs(x1 - x2) < 0.1:
        return

    # 3. Apply the new view boundaries
    # Using min/max ensures the zoom works regardless of drag direction (TL->BR or BR->TL)
    ax.set_xlim(min(x1, x2), max(x1, x2))
    ax.set_ylim(min(y1, y2), max(y1, y2))
    
    # This clears the yellow box graphics from the screen
    rect_selector.clear()
    
    # 4. Refresh & Feedback
    canvas.draw_idle()
    show_window_toast("Zoomed to Selection")

def on_press(event):
    global is_dragging, press_x, press_y
    
    # Safety Check: Ignore clicks on the toolbar or axis labels
    if event.inaxes != ax: 
        return
    
    # 1. Right-Click (Button 3): Initiate Panning
    if event.button == 3: 
        is_dragging = True
        # We store pixel coordinates (event.x) for smoother math during motion
        press_x, press_y = event.x, event.y

    # 2. Left Double-Click: Trigger Analysis
    elif event.dblclick and event.button == 1:
        if event.xdata is not None:
            analysis_type(event.xdata)
            
    # If middle-click (Scroll wheel click), reset view
    if event.button == 2: 
        reset_zoom()
        
def on_motion(event):
    global press_x, press_y
    
    # 1. Early Exit Logic
    if not is_dragging or event.inaxes != ax: 
        return
    if event.x is None or event.y is None: 
        return

    # 2. Calculate Distance Moved in Pixels
    dx = event.x - press_x
    dy = event.y - press_y
    
    # Update anchor for the next frame
    press_x, press_y = event.x, event.y

    # 3. Convert Pixels to Data Units
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    bbox = ax.get_window_extent()
    
    shift_x = (dx / bbox.width) * (cur_xlim[1] - cur_xlim[0])
    shift_y = (dy / bbox.height) * (cur_ylim[1] - cur_ylim[0])

    # 4. Update Axis Limits (The actual 'Pan')
    ax.set_xlim(cur_xlim[0] - shift_x, cur_xlim[1] - shift_x)
    ax.set_ylim(cur_ylim[0] - shift_y, cur_ylim[1] - shift_y)

    # 5. Fast Redraw
    # Because we use Blended Transforms for notes, they move WITH the axis.
    # No more deleting/re-adding in a loop!
    canvas.draw_idle()
    
def on_release(event):
    global is_dragging
    is_dragging = False

def launch_zscore_peth(center_t):
    if cache is None: return
    
    # 1. ANALYSIS PIPELINE (Delegated to Library)
    data_source = cache['corr'] if show_corrected else cache['raw']
    clean_signal = ProcessingLibrary.smooth_signal(data_source, cache['fs'])
    
    slice_x, z_seg = ProcessingLibrary.get_zscore_slice(cache['x'], clean_signal, center_t, window=30)
    
    if z_seg is not None:
        z_binned = ProcessingLibrary.bin_for_heatmap(z_seg)
        mode_str = "Corrected" if show_corrected else "Raw"

        # 2. UI SETUP (Toplevel Window)
        pop = tk.Toplevel(root)
        pop.title(f"PETH Analysis ({mode_str}) - {center_t:.2f}s")
        
        fig_peth = Figure(figsize=(8, 7), dpi=100)
        ax_heat, ax_line = fig_peth.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
        
        # 3. HEATMAP PLOT
        ax_heat.imshow(z_binned.reshape(1, -1), aspect='auto', cmap='YlGnBu_r', 
                       extent=[-30, 30, 0, 1], vmin=-5, vmax=5, interpolation='bilinear') 
        ax_heat.set_yticks([]) 
        ax_heat.set_ylabel("Intensity", fontweight='bold')
        
        # 4. LINE PLOT
        ax_line.plot(slice_x - center_t, z_seg, color='black', linewidth=1.5) 
        ax_line.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax_line.set_xlim([-30, 30])
        ax_line.set_ylim([-5, 5])
        ax_line.set_ylabel(f"Z-Score ({mode_str})", fontweight='bold')
        ax_line.set_xlabel("Time from Center (s)", fontweight='bold')
        
        fig_peth.suptitle(f"Z-score Peth ({mode_str} Data)", fontsize=14, fontweight='bold')
        fig_peth.tight_layout(rect=[0, 0.05, 1, 0.95]) 
        
        canvas_peth = FigureCanvasTkAgg(fig_peth, master=pop)
        canvas_peth.get_tk_widget().pack(fill="both", expand=True)

        # 5. EXPORT LOGIC
        def save_peth_action():
            ts = datetime.datetime.now().strftime("%H%M%S")
            default_fn = f"PETH_{mode_str}_{int(center_t)}s_{ts}.png"
            
            fpath = filedialog.asksaveasfilename(
                initialdir=os.path.dirname(folder_path) if folder_path else os.getcwd(),
                defaultextension=".png",
                initialfile=default_fn,
                title="Export PETH Analysis"
            )

            if fpath:
                try:
                    # CRUCIAL: Save 'fig_peth', NOT the global 'fig'
                    fig_peth.savefig(fpath, dpi=300, bbox_inches='tight')
                    show_window_toast("✅ PETH Exported Successfully")
                except Exception as e:
                    show_error(f"Export Failed: {str(e)}")

        btn_frame = tk.Frame(pop)
        btn_frame.pack(side="bottom", fill="x", pady=10)
        
        tk.Button(btn_frame, text=f"💾 Export {mode_str} PETH", 
                  command=save_peth_action, bg="#2196F3", fg="white", 
                  font=('Helvetica', 10, 'bold'), padx=20).pack()
        
        # Confirmation Toast
        show_window_toast(f"PETH Generated at {center_t:.1f}s")
        
def zoom_factory(ax, base_scale=1.2):
    def zoom_fun(event):
        # Safety check: ignore scroll events on the toolbar/buttons
        if event.x is None or event.y is None: return
        
        bbox = ax.get_window_extent()
        
        # 1. Zone Detection (The UX Secret Sauce)
        is_on_x_axis = event.y < bbox.ymin  
        is_on_y_axis = event.x < bbox.xmin  
        is_inside = event.inaxes == ax      

        # 2. Scale Calculation
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()

        # 3. Mode A: X-Only Zoom (User scrolls on the X-Axis labels)
        if is_on_x_axis and not is_on_y_axis:
            # Use 50/50 anchor if mouse isn't over a specific data point
            xdata = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])

        # 4. Mode B: Y-Only Zoom (User scrolls on the Y-Axis labels)
        elif is_on_y_axis and not is_on_x_axis:
            ydata = event.ydata if event.ydata is not None else (cur_ylim[0] + cur_ylim[1]) / 2
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])

        # 5. Mode C: Uniform Zoom (User scrolls in the data area)
        elif is_inside and event.xdata is not None and event.ydata is not None:
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
    """
    Used only for critical/blocking errors. 
    In a literature review, this is 'Modal Error Handling' 
    reserved for state-breaking events.
    """
    tk.messagebox.showerror("NeuroData Error", f"❌ {msg}")

# FOR SUCCESS (Blue 'i' icon)
def show_success(msg):
    """
    Replaces blocking popups with non-intrusive toast notifications.
    This allows the user to continue interacting with data without 
    interruption (Non-blocking UI).
    """
    show_window_toast(f"✅ {msg}")

def choose_file():
    global folder_path
    path = filedialog.askdirectory()
    
    # Validation logic is now 'black-boxed' in the library (Good for science!)
    is_valid, result = ProcessingLibrary.validate_tdt_folder(path)
    
    if is_valid:
        folder_path = path
        root.title(f"NeuroData Interface - {result}")
        # Using show_success here, which we've mapped to our new Toast
        show_success(f"Linked to: {result}")
    else:
        # Errors still use popups because they require immediate attention
        show_error(result)
    
#LOADING THE DATA INTO THE GUI
def load_data_action():
    global cache
    if folder_path is None:
        show_error("Please select a folder first!")
        return

    try:
        # One call to the library handles the entire science pipeline
        cache = ProcessingLibrary.process_tdt_folder(folder_path)
        
        msg = f"Loaded {cache['store']} ({cache['fs']:.1f} Hz)"
        show_success(msg)
        
        # Trigger the initial plot
        simple_plot() 
        
    except Exception as e:
        show_error(f"Processing Failed: {str(e)}")

#LOADING THE DATA IN THE GUI
def load_data_set():
    """
    The 'Master Trigger' for the data pipeline.
    Combines folder selection and automated processing into one action.
    """
    # 1. Open the dialog (Updates 'folder_path' if successful)
    choose_file()
    
    # 2. Only proceed to load if the user didn't cancel the dialog
    if folder_path:
        load_data_action()

#ADDS NOTES TO THE PLOT
def _update_plot_with_notes(markers):
    import matplotlib.transforms as transforms
    # X=Data coordinates, Y=0to1 (Axes fraction)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    unique_labels = set()
    
    # We remove the xmin/xmax filter here because 'clip_on=True' 
    # handles this at the C++ level in Matplotlib, which is much faster.
    for m in markers:
        # Manage legend redundancy
        label_id = m['label'] if m['label'] not in unique_labels else "_nolegend_"
        unique_labels.add(m['label'])
        
        # Draw the marker line
        ax.axvline(x=m['time'], color=m['color'], linestyle='--', alpha=0.3, label=label_id)
        
        # Draw the label using the sticky transform
        ax.text(m['time'], 0.98, f" {m['label']}", 
                transform=trans, 
                rotation=90, 
                va='top',          # Vertical Alignment
                clip_on=True,      # Automatically hides if out of view
                fontsize=8, 
                color=m['color'], 
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

#PLOT CORRECTED FOR BLEACHING        
def simple_plot(draw_now=True):
    # 1. Safety Check: Use the clean 'is None' check
    if cache is None: return
    
    
    # 2. Reset the canvas
    ax.clear()
    
    # 3. Pull from Cache (Zero calculation here = High Performance)
    data_to_plot = cache['corr'] if show_corrected else cache['raw']
    label_text = "Debleached Signal" if show_corrected else "Raw Fluorescence"
    color_choice = "blue" if show_corrected else "gray" # Blue vs Gray

    # Horizontal line at 0 (Amplitude)
    ax.axhline(0, color='black', linewidth=1.2, alpha=0.5, zorder=1)
    # Vertical line at 0 (Time start)
    ax.axvline(0, color='black', linewidth=1.2, alpha=0.5, zorder=1)
    
    
    # 5. The Main Trace
    ax.plot(cache['x'], data_to_plot, color=color_choice, lw=1, alpha=0.8, label=label_text)
    
    # 6. Metadata Overlay
    _update_plot_with_notes(cache['markers'])
    
    # 7. Aesthetics & Legend
    ax.set_title(f"{label_text} - {cache['store']}", fontweight='bold', pad=15)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")
    
    # Only draw if we aren't mid-toggle
    if draw_now:
        canvas.draw()

def export_canvas_action():
    # Only need cache; 'fig' is already accessible in the global scope
    if cache is None:
        show_error("No data loaded to export!")
        return

    # 1. Automated Naming (YearMonthDay_HourMinute)
    # This prevents the user from accidentally overwriting old files—a major UX win.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    initial_name = f"{cache['store']}_Plot_{timestamp}.png"
    
    # 2. Multi-Format Save Dialog
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG Image (Standard)", "*.png"), 
            ("PDF Document (Vector)", "*.pdf"), 
            ("SVG Vector (Editable)", "*.svg")
        ],
        initialfile=initial_name,
        title="Export Current View"
    )

    if file_path:
        try:
            # 3. High-Fidelity Rendering
            # 'bbox_inches=tight' ensures no labels are cut off at the edges
            fig.savefig(file_path, dpi=300, bbox_inches='tight', transparent=False)
            
            # Non-blocking feedback
            show_window_toast(f"✅ Exported: {os.path.basename(file_path)}")
            
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
plot_type_var.set("Analysis")
plot_options = ["Z-Score PETH"]
dropdown = tk.OptionMenu(options_frame, plot_type_var, *plot_options)
dropdown.pack(side="left", padx=10)

# --- EXECUTE ---
def analysis_type(data):
    choice = plot_type_var.get()
    if choice == "Z-Score PETH":
        launch_zscore_peth(data)


# ---  Zoom Reset ---
def reset_zoom():
    if 'cache' not in globals(): return

    # 1. Clear the graph
    ax.clear()
    simple_plot()
    
    
    # 2. Re-link the rectangle selector so it doesn't break
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