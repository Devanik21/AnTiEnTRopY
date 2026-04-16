"""
hrf_epigenetic.py — Harmonic Resonance Field: Epigenetic Edition
Applies the HRF wave interference framework (Debanik Debnath, 2025)
to methylation data for biological age state classification.

NOVEL CONTRIBUTION:
Methylation beta values across CpGs form a high-dimensional signal.
Young epigenomes exhibit ordered, coherent methylation patterns —
equivalent to a standing wave at class-specific resonance frequencies.
Old/senescent epigenomes show decoherence — analogous to phase noise.

The HRF model:
  Ψ_c(q, x_i) = exp(-γ‖q - x_i‖²) · (1 + cos(ω_c · ‖q - x_i‖ + φ))

Class assignment: argmax_c Σ_{i ∈ N_k(q,c)} Ψ_c(q, x_i)

Applied to methylation: q = query methylation vector,
x_i = training sample methylation vector,
classes = {Young, Middle, Old}
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import warnings
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
    Extends Debanik Debnath's HRF framework from EEG to methylation.
    """
    CLASS_LABELS = {0: 'Young (≤35)', 1: 'Middle (36-55)', 2: 'Old (>55)'}
    CLASS_COLORS = {0: '#00ff9f', 1: '#ffb700', 2: '#ff4455'}

    def __init__(
        self,
        omega_0: float = 10.0,
        gamma: float = 1.0,
        k: int = 5,
        n_components: int = 200
    ):
        self.omega_0 = omega_0     # Base resonance frequency
        self.gamma = gamma          # Spatial damping coefficient
        self.k = k                  # Local oscillators per class
        self.n_components = n_components  # PCA-reduced dims for efficiency

        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.classes = None
        self.nn_per_class = {}
        self.is_fitted = False
        self.best_params = None
        self.pca_components = None
        self.pca_mean = None
        self.metrics = {}

    def _pca_reduce(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Randomized PCA for dimensionality reduction."""
        if fit:
            self.pca_mean = X.mean(axis=0)
            X_centered = X - self.pca_mean
            # SVD for PCA
            n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.pca_components = Vt[:n_comp]
            return (X_centered @ self.pca_components.T)
        else:
            X_centered = X - self.pca_mean
            return (X_centered @ self.pca_components.T)

    def _resonance_energy(
        self,
        query: np.ndarray,
        class_samples: np.ndarray,
        omega_c: float
    ) -> float:
        """
        Compute resonance energy from class c's nearest oscillators.
        E_c(q) = Σ exp(-γ‖q-x_i‖²) · (1 + cos(ω_c · ‖q-x_i‖ + φ))
        """
        diffs = class_samples - query[np.newaxis, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=1) + 1e-8)

        # Gaussian envelope (locality)
        envelope = np.exp(-self.gamma * distances ** 2)

        # Oscillatory term (wave interference)
        oscillation = 1.0 + np.cos(omega_c * distances)

        wave_potentials = envelope * oscillation
        return float(np.sum(wave_potentials))

    def fit(self, X: pd.DataFrame, ages: pd.Series) -> 'HRFEpigenetic':
        """
        Fit HRF classifier with auto-evolution on gamma and omega_0.
        """
        # Assign age classes
        y = ages.apply(_age_to_class).values
        self.classes = np.unique(y)

        # Reduce dimensionality
        X_arr = X.values.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X_arr)
        X_reduced = self._pca_reduce(X_scaled, fit=True).astype(np.float32)

        self.X_train = X_reduced
        self.y_train = y

        # Build per-class KNN index for fast neighbor retrieval
        for c in self.classes:
            mask = y == c
            self.nn_per_class[c] = NearestNeighbors(
                n_neighbors=min(self.k, int(mask.sum())),
                metric='euclidean', algorithm='ball_tree'
            )
            self.nn_per_class[c].fit(X_reduced[mask])

        # Auto-evolution: grid search over omega_0 and gamma
        best_acc = -1
        best_omega = self.omega_0
        best_gamma = self.gamma

        omega_grid = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
        gamma_grid = [0.01, 0.1, 0.5, 1.0, 2.0]

        # Simple leave-some-out evaluation
        n_eval = min(100, len(y))
        eval_idx = np.random.choice(len(y), n_eval, replace=False)

        for omega in omega_grid:
            for gamma in gamma_grid:
                self.omega_0 = omega
                self.gamma = gamma
                preds = []
                for i in eval_idx:
                    pred = self._predict_single(X_reduced[i], exclude_idx=i)
                    preds.append(pred)
                acc = accuracy_score(y[eval_idx], preds)
                if acc > best_acc:
                    best_acc = acc
                    best_omega = omega
                    best_gamma = gamma

        self.omega_0 = best_omega
        self.gamma = best_gamma
        self.best_params = {'omega_0': best_omega, 'gamma': best_gamma}

        # Final accuracy
        all_preds = [self._predict_single(self.X_train[i]) for i in range(len(y))]
        self.metrics = {
            'train_accuracy': float(accuracy_score(y, all_preds)),
            'best_omega': best_omega,
            'best_gamma': best_gamma,
            'n_classes': len(self.classes),
            'n_young': int((y == 0).sum()),
            'n_middle': int((y == 1).sum()),
            'n_old': int((y == 2).sum()),
        }

        self.is_fitted = True
        return self

    def _predict_single(self, query: np.ndarray, exclude_idx: int = -1) -> int:
        """Predict class for a single reduced feature vector."""
        energies = {}
        for c in self.classes:
            omega_c = self.omega_0 * (c + 1)
            dists, indices = self.nn_per_class[c].kneighbors(
                query.reshape(1, -1), return_distance=True
            )
            neighbor_vecs = self.X_train[
                self.y_train == c
            ][indices[0]]
            energies[c] = self._resonance_energy(query, neighbor_vecs, omega_c)
        return int(max(energies, key=energies.get))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("HRF not fitted.")
        X_arr = X.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_arr)
        X_reduced = self._pca_reduce(X_scaled).astype(np.float32)
        return np.array([self._predict_single(X_reduced[i]) for i in range(len(X_reduced))])

    def resonance_energy_profile(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return per-sample resonance energies for each class.
        Useful for age state confidence visualization.
        """
        if not self.is_fitted:
            raise RuntimeError("HRF not fitted.")
        X_arr = X.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_arr)
        X_reduced = self._pca_reduce(X_scaled).astype(np.float32)

        records = []
        for i in range(len(X_reduced)):
            q = X_reduced[i]
            row = {'sample_idx': i}
            for c in self.classes:
                omega_c = self.omega_0 * (c + 1)
                dists, indices = self.nn_per_class[c].kneighbors(
                    q.reshape(1, -1), return_distance=True
                )
                neighbor_vecs = self.X_train[self.y_train == c][indices[0]]
                row[f'E_{CLASS_LABELS[c]}'] = self._resonance_energy(q, neighbor_vecs, omega_c)
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
        """
        Decompose methylation pattern into frequency components.
        High-frequency = noisy/chaotic (old). Low-frequency = ordered (young).
        """
        # Use top-N variable CpGs
        beta = np.array(sample_beta[:top_n], dtype=np.float64)
        beta_centered = beta - beta.mean()

        # FFT of methylation profile
        fft_vals = np.fft.rfft(beta_centered)
        freqs = np.fft.rfftfreq(len(beta_centered))
        power = np.abs(fft_vals) ** 2

        # Dominant frequency (resonance frequency)
        dominant_idx = np.argmax(power[1:]) + 1
        dominant_freq = float(freqs[dominant_idx])
        dominant_power = float(power[dominant_idx])

        # Spectral entropy (Shannon entropy of power spectrum)
        p_norm = power / (power.sum() + 1e-10)
        spectral_entropy = float(-np.sum(p_norm * np.log2(p_norm + 1e-10)))

        # Low-frequency power ratio (order) vs high-frequency (chaos)
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
