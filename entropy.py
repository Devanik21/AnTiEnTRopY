"""
entropy.py — Epigenetic Entropy Engine
Computes methylation disorder from CpG beta values.

Core principle: beta = 0 or 1 → ordered (young)
               beta = 0.5    → maximum disorder (aging)

Binary entropy H(β) = -β·log2(β) - (1-β)·log2(1-β)
peaks at β = 0.5 (H=1.0) and reaches 0 at β ∈ {0,1}.

Biological interpretation: Aging = epigenetic drift toward stochastic
methylation (0.5), eroding the deterministic methylation landscape.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

_EPS = 1e-10


def _binary_entropy(beta: np.ndarray) -> np.ndarray:
    """
    Compute per-site binary entropy H(β).
    H(β) = -β·log2(β) - (1-β)·log2(1-β)
    Returns 0 for β ∈ {0,1}, 1 for β = 0.5
    """
    b = np.clip(beta.astype(np.float64), _EPS, 1.0 - _EPS)
    return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))


def _methylation_order(beta: np.ndarray) -> float:
    """
    Methylation Order Index (MOI): 1 - mean(H(β))
    MOI = 1 → perfectly ordered (young)
    MOI = 0 → maximum disorder (senescent)
    """
    h = _binary_entropy(beta)
    return float(1.0 - np.mean(h))


class EpigeneticEntropy:
    """
    Full epigenetic entropy analysis pipeline.
    Computes disorder landscape across samples and CpGs.
    """

    def __init__(self):
        self.sample_entropy = None
        self.cpg_entropy_stats = None
        self.age_entropy_fit = None
        self.drift_cpgs = None
        self.is_computed = False

    def compute(self, X: pd.DataFrame, ages: pd.Series) -> 'EpigeneticEntropy':
        """
        Full entropy analysis.
        X: samples × CpGs methylation beta values
        ages: chronological ages
        """
        X_arr = X.values.astype(np.float64)
        ages_arr = ages.values.astype(np.float64)

        # 1. Per-sample entropy profile
        print("Computing per-sample entropy...")
        sample_h = np.apply_along_axis(_binary_entropy, 1, X_arr)  # (n_samples, n_cpgs)
        mean_h = sample_h.mean(axis=1)
        moi = 1.0 - mean_h

        # Fraction of CpGs near maximum disorder (β within 0.1 of 0.5)
        near_chaos = ((X_arr > 0.4) & (X_arr < 0.6)).mean(axis=1)

        # Fraction fully methylated/unmethylated (ordered states)
        fully_methylated = (X_arr > 0.8).mean(axis=1)
        fully_unmethylated = (X_arr < 0.2).mean(axis=1)
        ordered_fraction = fully_methylated + fully_unmethylated

        self.sample_entropy = pd.DataFrame({
            'sample_id': X.index,
            'chronological_age': ages_arr,
            'mean_entropy': mean_h.astype(np.float32),
            'methylation_order_index': moi.astype(np.float32),
            'chaos_fraction': near_chaos.astype(np.float32),
            'ordered_fraction': ordered_fraction.astype(np.float32),
            'fully_methylated_frac': fully_methylated.astype(np.float32),
            'fully_unmethylated_frac': fully_unmethylated.astype(np.float32),
        })

        # 2. Age-entropy correlation
        r, p = pearsonr(ages_arr, mean_h)
        slope, intercept, _, _, _ = stats.linregress(ages_arr, mean_h)
        self.age_entropy_fit = {
            'pearson_r': float(r),
            'p_value': float(p),
            'slope': float(slope),        # entropy increase per year
            'intercept': float(intercept),
            'entropy_per_decade': float(slope * 10),
        }

        # 3. Per-CpG drift analysis (correlation of beta with age)
        print("Computing CpG-level drift (this may take a moment)...")
        n_cpgs = X_arr.shape[1]

        # Vectorized correlation: (X - mean_X) dot (ages - mean_ages)
        X_centered = X_arr - X_arr.mean(axis=0, keepdims=True)
        ages_centered = ages_arr - ages_arr.mean()

        numerator = X_centered.T @ ages_centered  # (n_cpgs,)
        denom_x = np.sqrt((X_centered ** 2).sum(axis=0))
        denom_a = np.sqrt((ages_centered ** 2).sum())
        with np.errstate(invalid='ignore', divide='ignore'):
            cpg_r = np.where(denom_x > 0, numerator / (denom_x * denom_a), 0.0)

        # Per-CpG mean entropy
        cpg_mean_h = _binary_entropy(X_arr).mean(axis=0)

        # Per-CpG variance
        cpg_var = X_arr.var(axis=0)

        self.cpg_entropy_stats = pd.DataFrame({
            'cpg': X.columns,
            'mean_beta': X_arr.mean(axis=0).astype(np.float32),
            'std_beta': X_arr.std(axis=0).astype(np.float32),
            'mean_entropy': cpg_mean_h.astype(np.float32),
            'age_correlation': cpg_r.astype(np.float32),
            'variance': cpg_var.astype(np.float32),
        })
        self.cpg_entropy_stats['drift_type'] = self.cpg_entropy_stats['age_correlation'].apply(
            lambda r: 'Hypermethylated' if r > 0.2 else ('Hypomethylated' if r < -0.2 else 'Stable')
        )

        # 4. High-drift CpGs (|r| > 0.3)
        self.drift_cpgs = self.cpg_entropy_stats[
            self.cpg_entropy_stats['age_correlation'].abs() > 0.3
        ].sort_values('age_correlation', ascending=False)

        self.is_computed = True
        return self

    def get_entropy_trajectory(self, n_bins: int = 10) -> pd.DataFrame:
        """Age-binned entropy for trajectory visualization."""
        if not self.is_computed:
            raise RuntimeError("Run compute() first.")
        df = self.sample_entropy.copy()
        df['age_bin'] = pd.cut(df['chronological_age'], bins=n_bins)
        trajectory = df.groupby('age_bin').agg(
            mean_entropy=('mean_entropy', 'mean'),
            std_entropy=('mean_entropy', 'std'),
            mean_moi=('methylation_order_index', 'mean'),
            mean_chaos=('chaos_fraction', 'mean'),
            n_samples=('mean_entropy', 'count'),
            age_mid=('chronological_age', 'mean')
        ).reset_index()
        return trajectory.dropna()

    def get_entropy_summary(self) -> dict:
        if not self.is_computed:
            raise RuntimeError("Run compute() first.")
        return {
            'mean_entropy_young': float(
                self.sample_entropy[
                    self.sample_entropy['chronological_age'] <
                    self.sample_entropy['chronological_age'].quantile(0.25)
                ]['mean_entropy'].mean()
            ),
            'mean_entropy_old': float(
                self.sample_entropy[
                    self.sample_entropy['chronological_age'] >
                    self.sample_entropy['chronological_age'].quantile(0.75)
                ]['mean_entropy'].mean()
            ),
            'n_drift_cpgs': len(self.drift_cpgs),
            'n_hyper': int((self.drift_cpgs['age_correlation'] > 0.3).sum()),
            'n_hypo': int((self.drift_cpgs['age_correlation'] < -0.3).sum()),
            **self.age_entropy_fit
        }

    def get_sample_entropy_at(self, beta_values: np.ndarray) -> dict:
        """Compute entropy metrics for a single sample's beta values."""
        h = _binary_entropy(beta_values)
        moi = 1.0 - np.mean(h)
        chaos = float(np.mean((beta_values > 0.4) & (beta_values < 0.6)))
        ordered = float(np.mean((beta_values > 0.8) | (beta_values < 0.2)))
        return {
            'mean_entropy': float(np.mean(h)),
            'methylation_order_index': float(moi),
            'chaos_fraction': chaos,
            'ordered_fraction': ordered,
            'entropy_distribution': h
        }
