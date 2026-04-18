# System Architecture

## Core Modules
1. **`AnTiEnTRopY.py`**: The main Streamlit orchestrator. Handles file uploads, parameters, and tab rendering.
2. **`EnTRopY.py`**: Calculates Shannon entropy, Methylation Order Index (MOI), and identifies drifting CpGs.
3. **`CloCk.py`**: Implements an ElasticNet-based biological age predictor.
4. **`HRF_EpIgEnEtIc.py`**: A novel classifier using a Harmonic Resonance Field (HRF) approach and Fast Fourier Transform for spectral analysis.
5. **`ReVeRsAL.py`**: Simulates partial epigenetic reprogramming by shifting high-drift CpGs towards young reference values.
6. **`ImMoRtAlItY.py`**: Uses Monte Carlo simulations to model the conditions for "epigenetic escape velocity."

## Data Flow
Upload -> Preprocessing -> Clock Training -> Entropy Calculation -> HRF Classification -> Reversal Simulation -> Monte Carlo Projections -> Research Report Generation.
