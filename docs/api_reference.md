# API Reference

This document outlines the core classes and functions in the AnTiEnTRopY library.

## EpigeneticEntropy (`EnTRopY.py`)
- `__init__(self, eps=1e-10)`
- `calculate_entropy(self, beta_matrix)`
- `calculate_moi(self, beta_matrix)`
- `identify_drift_cpgs(self, beta_matrix, ages)`

## BiologicalClock (`CloCk.py`)
- `__init__(self, top_k=5000)`
- `fit(self, beta_matrix, ages)`
- `predict(self, beta_matrix)`
- `calculate_ieaa(self, predicted_ages, chrono_ages)`

## HRFEpigenetic (`HRF_EpIgEnEtIc.py`)
- `__init__(self, d_components=200)`
- `fit(self, beta_matrix, age_classes)`
- `predict_resonance(self, query_vector)`
- `spectral_decomposition(self, beta_vector)`

## ReversalSimulator (`ReVeRsAL.py`)
- `__init__(self, young_ref, old_ref)`
- `simulate_intervention(self, beta_matrix, p_int)`
- `generate_reversal_curve(self, max_p_int)`

## ImmortalityEngine (`ImMoRtAlItY.py`)
- `__init__(self, entropy_rate)`
- `calculate_escape_velocity(self, reversal_curve, T_interval)`
- `run_monte_carlo(self, initial_age, n_steps)`
