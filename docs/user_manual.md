# User Manual

## Overview
AnTiEnTRopY is an interactive Streamlit application. It takes a CSV of DNA methylation beta values and chronological ages, and outputs a suite of analyses across 6 main tabs.

## Preparing Data
Your input CSV must contain:
1. An `age` column (chronological age in years).
2. CpG site columns (e.g., `cg00000029`), where values are between 0 and 1.

## Navigation
1. **Upload Data:** Use the sidebar to upload your CSV file.
2. **Parameters:** Adjust parameters such as top-K CpGs for the clock or intervention percentage.
3. **Tabs:** Navigate through Biological Clock, Epigenetic Entropy, HRF Classifier, Reversal Simulator, Immortality Engine, and the Research Report to view different analyses.
