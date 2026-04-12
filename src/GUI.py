"""
NeuroData Interface (GUI)

Interactive Tkinter + Matplotlib application for fiber photometry analysis.

This GUI provides:
- TDT data loading and visualization
- Real-time interactive signal exploration
- Motion-corrected vs raw signal toggling
- Peri-event time histogram (PETH) analysis
- Event-aligned neuroscience data inspection
- Export of publication-quality figures

Backend processing is handled by ProcessingLibrary.py
which performs signal extraction, ΔF/F computation, filtering,
and photobleaching correction.
"""
import tkinter as tk
from tkinter import filedialog
import os
import ProcessingLibrary
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
    Displays a non-blocking toast notification in the bottom-right corner of the GUI.

    This function creates a temporary, borderless Tkinter window that provides
    lightweight user feedback without interrupting workflow (e.g., replacing modal popups).

    The toast automatically positions itself relative to the main application window
    and disappears after a specified duration.

    Parameters
    ----------
    message : str
        Text message displayed in the toast notification.
        Typically used for success, status updates, or soft warnings.

    duration : int, optional
        Time (in milliseconds) before the toast automatically closes.
        Default is 2500 ms (2.5 seconds).

    Behavior
    --------
    - Creates a topmost Tkinter Toplevel window
    - Removes window decorations (no title bar or buttons)
    - Positions itself at bottom-right of main window
    - Auto-destroys after timeout
    - Does not block user interaction

    Use Case
    --------
    Ideal for:
    - Success confirmations (e.g., "Export complete")
    - Background process updates
    - Non-critical user feedback

    Avoid using for:
    - Critical errors requiring user action
    - Input validation failures
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
    
def on_select(eclick, erelease):
    """
    Handles rectangle selection (box zoom) on the main signal plot.

    This function is triggered by Matplotlib's RectangleSelector when the user
    drags a selection box over the signal plot. It interprets the selected
    region and updates the axis limits to zoom into that area.

    Parameters
    ----------
    eclick : matplotlib.backend_bases.MouseEvent
        Mouse press event defining the first corner of the selection box.

    erelease : matplotlib.backend_bases.MouseEvent
        Mouse release event defining the opposite corner of the selection box.

    Behavior
    --------
    - Validates that both click and release coordinates are valid
    - Prevents accidental micro-zooms from small mouse movements
    - Updates x/y axis limits based on selected region
    - Clears the selection rectangle overlay
    - Refreshes the canvas without blocking execution
    - Displays a toast notification confirming zoom action

    Notes
    -----
    - Designed for interactive exploration of fiber photometry signals
    - Ignores invalid selections (e.g., clicks outside axes or tiny drags)
    - Works in conjunction with RectangleSelector for GUI-driven zooming
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
    """
    Handles mouse press events on the main Matplotlib canvas.

    This function defines the primary interaction logic for user input,
    including panning, event-triggered analysis, and view resetting.

    Parameters
    ----------
    event : matplotlib.backend_bases.MouseEvent
        Mouse event object containing information about the click position,
        button type, and whether the event occurred inside the plot axes.

    Behavior
    --------
    This function supports multiple interaction modes:

    1. Right-click drag (Button 3)
       - Activates panning mode
       - Stores initial cursor position for continuous motion tracking

    2. Left double-click (Button 1 + dblclick)
       - Triggers analysis at selected time point
       - Sends event time to analysis pipeline (e.g., PETH generation)

    3. Middle-click (Button 2)
       - Resets zoom and restores default view

    Safety Checks
    --------------
    - Ignores clicks outside the main plot axes
    - Prevents unintended interactions with UI elements or toolbar

    Notes
    -----
    - Works in coordination with `on_motion` and `on_release`
    - Forms part of the custom interactive navigation system
    - Critical component of GUI usability for signal exploration
    """
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
    """
    Handles mouse movement events for interactive panning of the signal plot.

    This function implements custom drag-based panning by tracking mouse movement
    in pixel space and converting it into data-space axis shifts.

    It works in conjunction with `on_press` and `on_release` to provide a
    smooth, continuous navigation system for the Matplotlib canvas.

    Parameters
    ----------
    event : matplotlib.backend_bases.MouseEvent
        Mouse motion event containing current cursor position in both pixel
        and data coordinates.

    Behavior
    --------
    - Activates only when panning mode is enabled (right-click drag)
    - Computes pixel displacement between current and previous cursor position
    - Converts pixel movement into data coordinate shifts using axis scaling
    - Updates x and y axis limits to create smooth panning motion
    - Redraws canvas efficiently using `draw_idle()` for performance

    Safety Checks
    --------------
    - Ignores motion if not in dragging mode
    - Ignores events outside the main plotting axes
    - Prevents null or invalid coordinate updates

    Notes
    -----
    - This is a custom implementation of Matplotlib panning
    - Provides smoother and more controllable behavior than default navigation tools
    - Optimized for real-time exploration of fiber photometry signals
    """
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
    """
    Handles mouse button release events for interactive panning.

    This function terminates the panning interaction initiated by `on_press`
    and updated via `on_motion`. It resets the internal dragging state so
    that subsequent mouse movements no longer modify the plot view.

    Parameters
    ----------
    event : matplotlib.backend_bases.MouseEvent
        Mouse release event triggered when a mouse button is released.

    Behavior
    --------
    - Disables active dragging mode
    - Resets internal panning state flag
    - Ensures smooth termination of interactive navigation
    - Prevents unintended continued plot movement

    Notes
    -----
    - Works as the final step in the custom panning interaction loop
    - Must remain lightweight to avoid input lag or UI delay
    - Paired with `on_press` and `on_motion` for full navigation control
    """
    global is_dragging
    is_dragging = False

def launch_zscore_peth(center_t):
    """
    Launches a peri-event time histogram (PETH) analysis window centered on a selected event time.

    This function extracts a time window around a user-selected event, processes the signal
    into a smoothed and z-scored representation, and visualizes both a heatmap-style binning
    and a peri-event trace in a separate Tkinter window.

    The analysis is used to quantify neural signal dynamics surrounding behavioral events.

    Parameters
    ----------
    center_t : float
        Time (in seconds) of the event around which the peri-event analysis is centered.

    Behavior
    --------
    - Extracts signal segment around `center_t` from cached dataset
    - Applies smoothing and optional correction (raw or ΔF/F signal)
    - Computes z-scored peri-event trace using baseline normalization
    - Generates two-panel visualization:
        1. Heatmap-style binned activity representation
        2. Continuous peri-event z-score trace
    - Opens a new Tkinter window with embedded Matplotlib figure
    - Provides export functionality for saving the PETH figure

    Output
    ------
    - Interactive PETH window
    - Exportable high-resolution figure (PNG/PDF/SVG)

    Notes
    -----
    - Requires preloaded dataset in global `cache`
    - Uses `ProcessingLibrary` for signal smoothing, slicing, and binning
    - Supports both raw and motion/bleaching corrected signals
    - Designed for exploratory behavioral neuroscience analysis
    """
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
        ax_line.set_xlim([-15, 15])
        ax_line.set_ylim([-5, 5])
        ax_line.set_ylabel(f"Z-Score ({mode_str})", fontweight='bold')
        ax_line.set_xlabel("Time from Center (s)", fontweight='bold')
        
        fig_peth.suptitle("Z-score Peth", fontsize=14, fontweight='bold')
        fig_peth.tight_layout(rect=[0, 0.05, 1, 0.95]) 
        
        canvas_peth = FigureCanvasTkAgg(fig_peth, master=pop)
        canvas_peth.get_tk_widget().pack(fill="both", expand=True)

        # 5. EXPORT LOGIC
       def save_peth_action():
    """
    Saves the currently displayed peri-event time histogram (PETH) figure to disk.

    This function allows the user to export the generated PETH visualization
    from the analysis window in multiple formats for external use, such as
    publications, presentations, or further analysis.

    Behavior
    --------
    - Opens a file dialog for user-defined save location
    - Automatically generates a default filename with timestamp and event metadata
    - Saves the current PETH Matplotlib figure (`fig_peth`) to disk
    - Supports multiple export formats (PNG, PDF, SVG)
    - Provides success or error feedback via GUI toast notifications

    Output Formats
    --------------
    - PNG: Raster image (default, publication-ready)
    - PDF: Vector format for papers
    - SVG: Editable vector graphics for Illustrator/Inkscape

    Notes
    -----
    - Requires an active PETH window (`fig_peth` must exist in scope)
    - Uses high-resolution export settings (dpi=300, tight bounding box)
    - Designed to preserve visual fidelity of scientific plots
    """
            ts = datetime.datetime.now().strftime("%H%M%S")
            default_fn = f"PETH_{mode_str}_{int(center_t)}s_{ts}.png"
            
            fpath = filedialog.asksaveasfilename(
                initialdir=os.path.dirname(folder_path) if folder_path else os.getcwd(),
                defaultextension=".png",
                filetypes=[
                    ("PNG Image (Standard)", "*.png"), 
                    ("PDF Document (Vector)", "*.pdf"), 
                    ("SVG Vector (Editable)", "*.svg")
                ],
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
    """
    Creates a custom scroll-wheel zoom handler for a Matplotlib axis.

    This function returns an event handler that enables advanced zooming behavior
    depending on cursor position, allowing axis-aware and data-centered zoom control.

    The zoom behavior adapts dynamically based on where the user scrolls:
    - X-axis region → horizontal zoom only
    - Y-axis region → vertical zoom only
    - Plot area → uniform 2D zoom

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object to apply zoom behavior to.

    base_scale : float, optional
        Scaling factor for zoom speed. Values > 1 increase zoom sensitivity.
        Default is 1.2.

    Returns
    -------
    function
        Event handler function compatible with Matplotlib's scroll_event system.

    Behavior
    --------
    - Listens to mouse scroll events
    - Detects cursor position relative to plot axes and canvas
    - Applies contextual zooming logic:
        * Scroll on data → zoom in/out both axes
        * Scroll on x-axis → zoom horizontally only
        * Scroll on y-axis → zoom vertically only
    - Maintains focus on cursor position during zoom
    - Redraws canvas efficiently after updates

    Notes
    -----
    - Provides more intuitive navigation than default Matplotlib zoom tools
    - Designed specifically for exploratory signal analysis (e.g., photometry traces)
    - Improves precision when inspecting transient neural events
    """
    def zoom_fun(event):
    """
    Handles scroll-wheel zoom interactions for the Matplotlib canvas.

    This function is dynamically generated by `zoom_factory` and attached
    to the figure's scroll_event. It implements context-aware zooming based
    on cursor position relative to the plot axes.

    Parameters
    ----------
    event : matplotlib.backend_bases.MouseEvent
        Scroll event containing cursor position, scroll direction, and axis context.

    Behavior
    --------
    The zoom behavior depends on cursor location:

    1. X-axis region
       - Zooms horizontally only
       - Preserves vertical scale

    2. Y-axis region
       - Zooms vertically only
       - Preserves horizontal scale

    3. Data region (inside plot)
       - Applies uniform zoom on both axes
       - Maintains focus around cursor position

    Zoom Direction
    --------------
    - Scroll up → zoom in (increase resolution)
    - Scroll down → zoom out (expand view)

    Implementation Details
    ----------------------
    - Computes axis limits before and after scaling
    - Uses cursor position as zoom anchor point
    - Applies proportional scaling to maintain focus stability
    - Redraws canvas using `draw_idle()` for performance

    Notes
    -----
    - Designed for high-resolution exploration of time-series neural signals
    - Provides more intuitive navigation than default Matplotlib toolbar zoom
    - Critical for inspecting transient photometry events
    """
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
    Displays a critical error dialog to the user using a modal popup.

    This function is used for blocking or high-severity errors that require
    immediate user attention and cannot be resolved silently or via toast notifications.

    Parameters
    ----------
    msg : str
        Error message to display to the user. Should describe the failure
        in clear, user-readable language.

    Behavior
    --------
    - Opens a modal Tkinter error dialog (blocking UI interaction)
    - Halts user workflow until dismissed
    - Used for fatal or invalid operation states (e.g., missing data, failed processing)
    - Distinct from non-blocking notifications (e.g., `show_window_toast`)

    Use Case
    --------
    Appropriate for:
    - Data loading failures
    - Invalid folder selection
    - Processing pipeline crashes
    - Missing required inputs

    Notes
    -----
    - This is a high-priority UX interruption mechanism
    - Should not be overused for minor or recoverable issues
    - Pairs with `show_success` / `show_window_toast` for UX hierarchy
    """
    tk.messagebox.showerror("NeuroData Error", f"❌ {msg}")

# FOR SUCCESS (Blue 'i' icon)
def show_success(msg):
    """
    Displays a non-blocking success notification to the user.

    This function provides lightweight positive feedback using a toast-style
    notification instead of a modal popup, allowing uninterrupted workflow
    during data exploration and analysis.

    Parameters
    ----------
    msg : str
        Success message to display. Should confirm completion of an action
        or successful execution of a process.

    Behavior
    --------
    - Calls `show_window_toast` to display a temporary notification
    - Does not block user interaction
    - Appears in bottom-right of the main application window
    - Automatically disappears after a short duration

    Use Case
    --------
    Appropriate for:
    - Successful data loading
    - Completion of processing pipelines
    - Successful exports (plots, PETH, figures)
    - Confirmation of user actions

    Notes
    -----
    - Part of the non-blocking UI feedback system
    - Designed to maintain workflow continuity
    - Complements `show_error` (blocking alerts)
    """
    show_window_toast(f"✅ {msg}")

def choose_file():
    """
    Opens a directory selection dialog and validates a TDT dataset folder.

    This function allows the user to select a Tucker-Davis Technologies (TDT)
    recording directory through a GUI file picker. The selected folder is then
    validated to ensure it contains a valid TDT block structure.

    Validation is performed using the backend function:
    `ProcessingLibrary.validate_tdt_folder()`.

    Behavior
    --------
    - Opens a Tkinter folder selection dialog
    - Checks whether the selected folder contains valid TDT data (.Tbk files)
    - If valid:
        * Updates global `folder_path`
        * Updates GUI window title with dataset name
        * Displays success notification
    - If invalid:
        * Displays error message to user
        * Prevents further processing

    Returns
    -------
    None

    Notes
    -----
    - This is the primary entry point for loading datasets into the GUI
    - Acts as a safety gate before any signal processing occurs
    - Ensures only valid TDT structures are passed to the processing pipeline
    """
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
    """
    Loads and processes the selected TDT dataset into the GUI.

    This function acts as the main bridge between the user-selected folder
    and the signal processing pipeline defined in `ProcessingLibrary`.

    It performs full dataset processing and stores the result in a global
    cache for fast interactive visualization.

    Behavior
    --------
    - Checks that a folder has been selected (`folder_path`)
    - Calls `ProcessingLibrary.process_tdt_folder()` to execute full pipeline:
        * Signal extraction (465/415 nm)
        * Motion correction
        * Photobleaching correction
        * ΔF/F computation
        * Filtering and event extraction
    - Stores processed output in global `cache`
    - Updates GUI with success notification
    - Triggers initial plot rendering via `simple_plot()`

    Raises
    ------
    Displays error dialog if:
    - No folder has been selected
    - Processing pipeline fails or throws an exception

    Returns
    -------
    None

    Notes
    -----
    - This is the primary activation function of the analysis pipeline
    - All downstream visualization depends on successful execution of this function
    - Designed to separate GUI logic from scientific processing logic
    """
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
    Master trigger for the full data loading and processing pipeline.

    This function orchestrates the complete workflow from user dataset selection
    to processed signal visualization in a single automated action.

    It combines:
    - Folder selection via GUI
    - Dataset validation
    - Full TDT processing pipeline execution
    - Initial visualization rendering

    Behavior
    --------
    - Opens folder selection dialog (`choose_file`)
    - Validates selected TDT directory
    - If valid:
        * Processes dataset using `load_data_action`
        * Stores results in global cache
        * Triggers initial plot rendering
    - If invalid or cancelled:
        * Stops execution safely without side effects

    Returns
    -------
    None

    Notes
    -----
    - Serves as the primary “one-click start” entry point for the application
    - Abstracts all pipeline complexity from the user
    - Designed for fast experimental workflow initialization
    """
    # 1. Open the dialog (Updates 'folder_path' if successful)
    choose_file()
    
    # 2. Only proceed to load if the user didn't cancel the dialog
    if folder_path:
        load_data_action()

#ADDS NOTES TO THE PLOT
def _update_plot_with_notes(markers):
    """
    Overlays behavioral event markers onto the main signal plot.

    This function renders time-aligned behavioral annotations on top of the
    photometry signal using vertical lines and labeled text markers. It is
    responsible for visually linking neural activity with experimental events.

    Parameters
    ----------
    markers : list of dict
        List of event markers, where each marker contains:
        - 'time' : float
            Event timestamp in seconds
        - 'label' : str
            Event name (e.g., behavioral condition)
        - 'color' : str
            Color used for visualization of the event

    Behavior
    --------
    - Draws vertical dashed lines at event timestamps
    - Adds text labels aligned to the top of the plot
    - Uses blended coordinate transforms for stable label positioning
    - Ensures labels remain fixed relative to axes during zoom/pan
    - Avoids duplicate legend entries for repeated event types

    Visualization Details
    ---------------------
    - Lines are semi-transparent to avoid obscuring signal traces
    - Labels are rotated vertically for compact time-axis display
    - Uses axis blending to anchor text in normalized y-space

    Notes
    -----
    - This function is purely visual (no data modification)
    - Critical for interpreting neural activity in behavioral context
    - Designed for real-time interactive plotting environments
    """
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
    """
    Renders the main time-series signal plot in the GUI.

    This function is the primary visualization engine of the application.
    It clears and redraws the Matplotlib canvas using cached processed data,
    including signal traces and behavioral event overlays.

    It supports toggling between raw and processed (corrected) signals and
    integrates event markers for behavioral interpretation.

    Parameters
    ----------
    draw_now : bool, optional
        If True, immediately redraws the canvas.
        If False, updates plot state without forcing render (useful for batching).

    Behavior
    --------
    - Clears current axis content
    - Selects signal type based on global state (`show_corrected`)
    - Plots time-series neural signal from cached dataset
    - Draws reference axes (x=0, y=0) for temporal alignment
    - Overlays behavioral event markers via `_update_plot_with_notes`
    - Updates axis labels, title, and legend
    - Renders updated figure to GUI canvas

    Data Sources
    ------------
    - Uses global `cache` object generated by `process_tdt_folder`
    - Signal input:
        * `cache['corr']` if corrected mode is enabled
        * `cache['raw']` otherwise
    - Event markers from `cache['markers']`

    Notes
    -----
    - This is the central rendering function of the GUI
    - All interactive updates ultimately route through this function
    - Designed for high-frequency redraw without recomputation overhead
    - Computational efficiency is achieved via preprocessed cached data
    """
    # 1. Safety Check: Use the clean 'is None' check
    if cache is None: return
    
    
    # 2. Reset the canvas
    ax.clear()
    
    # 3. Pull from Cache (Zero calculation here = High Performance)
    data_to_plot = cache['corr'] if show_corrected else cache['raw']
    label_text = "Signal" 
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
    """
    Exports the current main visualization canvas to an external file.

    This function allows the user to save the currently displayed Matplotlib
    figure from the main GUI window in high resolution for external use such as
    publications, presentations, or offline analysis.

    It supports multiple output formats and ensures that the exported figure
    preserves visual fidelity and layout integrity.

    Behavior
    --------
    - Checks that a dataset is loaded in the global cache
    - Automatically generates a timestamped default filename
    - Opens a file dialog for user-defined save location
    - Saves the current figure (`fig`) to disk using high-resolution settings
    - Provides non-blocking success or error feedback via GUI notifications

    Supported Formats
    -----------------
    - PNG : Raster image (default, publication-ready)
    - PDF : Vector format suitable for manuscripts
    - SVG : Editable vector format for design tools

    Export Details
    --------------
    - Uses high DPI (300) for publication-quality output
    - Applies tight bounding box to prevent label clipping
    - Preserves current zoom/pan state of the visualization

    Notes
    -----
    - This function exports the MAIN canvas (not analysis popups like PETH)
    - Designed for reproducible scientific figure generation
    - Critical for transitioning from exploratory analysis to reporting
    """
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
    """
    Routes user-selected analysis types to the appropriate processing pipeline.

    This function acts as a dispatcher between the GUI selection menu and the
    corresponding analysis routines. It determines which analysis to execute
    based on the current value of the GUI dropdown menu.

    Parameters
    ----------
    data : float
        Input event time or center time (in seconds) used as the reference
        point for time-locked analyses (e.g., PETH generation).

    Behavior
    --------
    - Reads the selected analysis type from the GUI dropdown (`plot_type_var`)
    - Matches selection to available analysis routines
    - Executes the corresponding analysis function
    - Currently supports:
        * "Z-Score PETH" → launches `launch_zscore_peth(data)`

    Notes
    -----
    - This is a routing/dispatch function (no direct computation)
    - Designed for extensibility (new analysis types can be added easily)
    - Centralizes analysis selection logic for the GUI
    """
    choice = plot_type_var.get()
    if choice == "Z-Score PETH":
        launch_zscore_peth(data)


# ---  Zoom Reset ---
def reset_zoom():
    """
    Resets the main plot view to its default zoom and layout state.

    This function restores the original visualization state of the GUI after
    user-driven zooming or panning interactions. It clears current axis limits
    and redraws the full dataset from cache.

    Behavior
    --------
    - Clears current Matplotlib axis
    - Re-renders full signal using `simple_plot()`
    - Restores default view limits (full time-series range)
    - Reattaches interactive selection tools if needed
    - Refreshes the GUI canvas

    Use Case
    --------
    - User wants to return to full dataset view after zooming
    - Resetting exploration state during analysis
    - Recovering from over-zoom or lost navigation context

    Notes
    -----
    - Does not modify underlying data or cache
    - Only affects visualization state
    - Works as a “view reset” utility for the interactive plotting system
    """
    if 'cache' not in globals(): return

    # 1. Clear the graph
    ax.clear()
    simple_plot()
    
    
    # 2. Re-link the rectangle selector so it doesn't break
    if 'rect_selector' in globals():
        rect_selector.ax = ax
        rect_selector.set_active(True)

    canvas.draw_idle()
        
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
