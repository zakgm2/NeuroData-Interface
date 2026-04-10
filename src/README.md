# NeuroData Interface 
### Open-Source Fiber Photometry Analysis for Behavioral Neuroscience

An open-source, platform-agnostic Python library and GUI designed for analyzing fiber photometry data. Developed at **Concordia University**, this tool provides a streamlined workflow for debleaching, signal processing, and Peri-Event Time Histogram (PETH) visualization.

---

##  Key Features
* **Platform Agnostic:** Fully functional on **macOS**, Windows, and Linux.
* **Double-Exponential Debleaching:** Implements a physical model of fluorophore decay to restore signal baseline.
* **Interactive PETH Generation:** Align neural data to behavioral markers (e.g., "Claps," "Sucrose") with 30s/30s windows.
* **Point-and-Click Alignment:** Double-click the continuous trace to manually anchor $t=0$ for custom event analysis.
* **Standardized Filtering:** Built-in 5Hz Butterworth low-pass filtering for publication-quality traces.
* **Open room for more analysis methods** 
---

##  Installation & Dependencies
This library requires Python 3.8+. To set up your environment:

```bash
pip install numpy matplotlib scipy tdt
```
