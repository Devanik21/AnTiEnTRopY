"""
hrf_epigenetic.py — Harmonic Resonance Field: Epigenetic Edition (v15.0 Golden Grid Core)
Applies the HRF v15.0 wave interference framework (CPU-adapted)
to methylation data for biological age state classification.

NOVEL CONTRIBUTION:
Methylation beta values across CpGs form a high-dimensional signal.
We apply Bipolar Montage (adjacent differences + coherence variance)
and map resonances using a Harmonic Kernel with a 2.5 exponent.

The HRF v15.0 model:
  Ψ_c(q, x_i) = exp(-γ‖q - x_i‖^(2.5)) · (1 + cos(ω · ‖q - x_i‖))

Class assignment: argmax_c Σ_{i ∈ N_k(q,c)} Ψ_c(q, x_i)
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Age class boundaries
AGE_YOUNG_MAX = 35
AGE_MIDDLE_MAX = 55
# > 55 = Old

def _age_to_class(age: float) -> int:
    if age <= AGE_YOUNG_MAX:
        return 0  # Young
    elif age <= AGE_MIDDLE_MAX:
        return 1  # Middle
    else:
        return 2  # Old

CLASS_LABELS = {0: 'Young (≤35)', 1: 'Middle (36-55)', 2: 'Old (>55)'}
CLASS_COLORS = {0: '#00ff9f', 1: '#ffb700', 2: '#ff4455'}

class HRFEpigenetic:
    """
    Harmonic Resonance Field classifier for epigenetic age state.
    Upgraded to v15.0 Golden Grid Evolutionary Logic (CPU backend).
    """

    def __init__(
        self,
        omega_0: float = 10.0,
        gamma: float = 0.5,
        k: int = 5,
        auto_evolve: bool = True
    ):
        self.omega_0 = omega_0
        self.gamma = gamma
        self.k = k
        self.auto_evolve = auto_evolve

        self.scaler = RobustScaler(quantile_range=(15.0, 85.0))
        self.X_train = None
        self.y_train = None
        self.classes = None
        self.nn_model = None
        
        self.is_fitted = False
        self.metrics = {}
        self.all_evolution_scores = []
        self.best_params = {}

    def _apply_bipolar_montage(self, X: np.ndarray) -> np.ndarray:
        """Feature engineering: Bipolar difference mapping and variance coherence."""
        X_clip = np.clip(X, -15, 15)
        diffs = []
        for i in range(X_clip.shape[1] - 1):
            diffs.append(X_clip[:, i] - X_clip[:, i + 1])
        coherence = np.var(X_clip, axis=1).reshape(-1, 1)
        if len(diffs) > 0:
            diff_feat = np.array(diffs).T
            return np.hstack([X_clip, diff_feat, coherence]).astype(np.float32)
        else:
            return np.hstack([X_clip, coherence]).astype(np.float32)

    def _simulate_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, freq: float, gamma: float, k: int) -> np.ndarray:
        # We compute exactly as v15 CPU version
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='ball_tree')
        knn.fit(X_train)
        dists, indices = knn.kneighbors(X_query)
        
        preds = []
        for i in range(len(X_query)):
            local_dists = dists[i]
            local_y = y_train[indices[i]]
            
            energies = np.zeros(len(self.classes))
            for c_idx, c in enumerate(self.classes):
                mask = (local_y == c)
                d = local_dists[mask]
                if len(d) > 0:
                    # Harmonic Kernel Application (exponent 2.5)
                    w = np.exp(-gamma * (d**2.5)) * (1.0 + np.cos(freq * d))
                    energies[c_idx] = np.sum(w)
                else:
                    energies[c_idx] = 0.0
            
            preds.append(self.classes[np.argmax(energies)])
        return np.array(preds)

    def fit(self, X: pd.DataFrame, ages: pd.Series) -> 'HRFEpigenetic':
        # Assign age classes
        y = ages.apply(_age_to_class).values
        self.classes = np.unique(y)

        # Preprocessing
        X_arr = X.values.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X_arr)
        
        # Golden Grid v15 uses bipolar montage
        self.X_train = self._apply_bipolar_montage(X_scaled)
        self.y_train = y

        if self.auto_evolve and len(self.X_train) > 1:
            # Create a localized validation split for evolution
            n_sub = len(self.X_train)
            X_tr, X_val, y_tr, y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.24, stratify=self.y_train, random_state=9
            )

            best_score = -1
            best_dna = (self.omega_0, self.gamma, self.k)
            self.all_evolution_scores = []
            
            # v15.0 Golden Grid
            golden_grid = [
                (28.0, 10.0, 2), (30.0, 10.0, 1), (30.0, 10.0, 2), (50.0, 15.0, 2),
                (22.0, 9.0, 2), (18.0, 7.5, 2), (14.0, 5.0, 3), (16.0, 5.5, 3),
                (29.0, 10.0, 2), (31.0, 10.5, 2), (32.0, 11.0, 2), (33.0, 11.5, 2),
                (27.0, 9.5, 2), (26.0, 9.0, 2), (35.0, 12.0, 2), (34.0, 11.8, 2),
                (50.0, 15.0, 1), (52.0, 16.0, 2), (55.0, 17.0, 2), (60.0, 20.0, 2),
                (45.0, 13.5, 2), (48.0, 14.5, 2), (58.0, 19.0, 2), (65.0, 22.0, 2),
                (80.0, 25.0, 1), (90.0, 30.0, 1), (100.0, 35.0, 1), (120.0, 40.0, 1),
                (75.0, 24.0, 1), (85.0, 28.0, 1), (95.0, 32.0, 1), (110.0, 38.0, 1)
            ]

            for freq, gamma, k in golden_grid:
                preds = self._simulate_predict(X_tr, y_tr, X_val, freq, gamma, k)
                score = accuracy_score(y_val, preds)
                self.all_evolution_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_dna = (freq, gamma, k)

            self.omega_0, self.gamma, self.k = best_dna
            
        self.best_params = {'best_freq': self.omega_0, 'best_gamma': self.gamma, 'best_k': self.k}
        
        # Fit master model for actual predict() / resonance profiles
        self.nn_model = NearestNeighbors(n_neighbors=self.k, metric='euclidean', algorithm='ball_tree')
        self.nn_model.fit(self.X_train)

        # Final accuracy over all data
        all_preds = self._simulate_predict(self.X_train, self.y_train, self.X_train, self.omega_0, self.gamma, self.k)
        unique_scores = sorted(list(set(self.all_evolution_scores)), reverse=True)
        top3_evolution_peaks = unique_scores[:3] if unique_scores else [0, 0, 0]
        
        self.metrics = {
            'train_accuracy': float(accuracy_score(self.y_train, all_preds)),
            'best_freq': self.omega_0,
            'best_gamma': self.gamma,
            'best_k': self.k,
            'n_classes': len(self.classes),
            'n_young': int((self.y_train == 0).sum()),
            'n_middle': int((self.y_train == 1).sum()),
            'n_old': int((self.y_train == 2).sum()),
            'top3_evolution_peaks': top3_evolution_peaks
        }

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("HRF not fitted.")
        X_arr = X.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_arr)
        X_holo = self._apply_bipolar_montage(X_scaled)
        return self._simulate_predict(self.X_train, self.y_train, X_holo, self.omega_0, self.gamma, self.k)

    def _resonance_energy(self, query_dist, query_y, query_idx: np.ndarray, target_c: int) -> float:
        # Local energy component computation
        mask = (query_y == target_c)
        d = query_dist[mask]
        if len(d) > 0:
            w = np.exp(-self.gamma * (d**2.5)) * (1.0 + np.cos(self.omega_0 * d))
            return float(np.sum(w))
        return 0.0

    def resonance_energy_profile(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("HRF not fitted.")
        X_arr = X.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_arr)
        X_holo = self._apply_bipolar_montage(X_scaled)

        dists, indices = self.nn_model.kneighbors(X_holo)
        
        records = []
        for i in range(len(X_holo)):
            row = {'sample_idx': i}
            local_dists = dists[i]
            local_y = self.y_train[indices[i]]
            
            for c in self.classes:
                row[f'E_{CLASS_LABELS[c]}'] = self._resonance_energy(local_dists, local_y, indices[i], c)
            records.append(row)
            
        df = pd.DataFrame(records)
        energy_cols = [c for c in df.columns if c.startswith('E_')]
        totals = df[energy_cols].sum(axis=1)
        for col in energy_cols:
            df[col.replace('E_', 'P_')] = df[col] / (totals + 1e-8)
        df['predicted_class'] = df[energy_cols].idxmax(axis=1).str.replace('E_', '')
        return df

    def get_methylation_wave_signature(
        self,
        sample_beta: np.ndarray,
        cpg_names: list,
        top_n: int = 500
    ) -> dict:
        # Wave signature is structurally unmodified
        beta = np.array(sample_beta[:top_n], dtype=np.float64)
        beta_centered = beta - beta.mean()
        
        fft_vals = np.fft.rfft(beta_centered)
        freqs = np.fft.rfftfreq(len(beta_centered))
        power = np.abs(fft_vals) ** 2
        
        dominant_idx = np.argmax(power[1:]) + 1
        dominant_freq = float(freqs[dominant_idx])
        dominant_power = float(power[dominant_idx])
        
        p_norm = power / (power.sum() + 1e-10)
        spectral_entropy = float(-np.sum(p_norm * np.log2(p_norm + 1e-10)))
        
        mid = len(freqs) // 2
        low_power = float(power[:mid].sum())
        high_power = float(power[mid:].sum())
        coherence_ratio = low_power / (low_power + high_power + 1e-10)
        
        return {
            'frequencies': freqs[:50].tolist(),
            'power_spectrum': power[:50].tolist(),
            'dominant_frequency': dominant_freq,
            'dominant_power': dominant_power,
            'spectral_entropy': spectral_entropy,
            'coherence_ratio': float(coherence_ratio),
            'low_freq_power': low_power,
            'high_freq_power': high_power,
        }
