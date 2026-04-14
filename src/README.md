# NeuroData Interface 
### Open-Source Fiber Photometry Analysis for Behavioral Neuroscience

An open-source, platform-agnostic Python library and GUI designed for analyzing fiber photometry data. Developed at **Concordia University**, this tool provides a streamlined workflow for debleaching, signal processing, and Peri-Event Time Histogram (PETH) visualization.

### Pipeline:

-> Motion Artifact Correction 

-> Photobleaching Correction (Double Exponential Model giving baseline) 
 
-> Normalization 
 
-> Denoising 


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
Follow these steps to set up the environment and run the analysis tool.

## 1. Install Anaconda
If you don't have it yet, download and install **Anaconda** or **Miniconda**:
* [Download Anaconda](https://anaconda.com)

## 2. Create the Environment
Open your **Anaconda Prompt** (search for it in the Start Menu) and run these commands one by one to create a dedicated environment and install the necessary libraries:

```bash
# Create a new environment named 'neuro'
conda create -n neuro python=3.11 -y

# Activate the environment
conda activate neuro

# Install core scientific packages via Conda
conda install scipy numpy matplotlib pandas -y

# Install Spyder (the IDE)
conda install spyder -y

# Install TDT-specific libraries (not included in Anaconda by default)
pip install tdt
```
