"""
reversal.py — Anti-Entropy Reversal Simulator
Models partial epigenetic reprogramming interventions.

Scientific basis:
- Yamanaka factor partial reprogramming (Ocampo et al. 2016, Sinclair lab)
- CpG sites drift from youthful set-points; reversal moves them back
- Biological age reduction is quantified via clock re-prediction
- Reversal curve: years reversed vs. % sites intervened

Intervention model:
    β_new[i] = β_old[i] + α * (β_young[i] - β_old[i])
where α ∈ [0,1] is the reprogramming efficiency for site i.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ReversalSimulator:
    """
    Simulates partial epigenetic reprogramming on individual samples.
    Uses a young reference methylome as the reprogramming target.
    """

    def __init__(self, young_percentile: float = 20.0):
        self.young_percentile = young_percentile
        self.young_reference = None
        self.old_reference = None
        self.drift_magnitude = None
        self.feature_names = None
        self.is_built = False

    def build_references(self, X: pd.DataFrame, ages: pd.Series) -> 'ReversalSimulator':
        """
        Compute young and old reference methylation profiles.
        young_reference: mean beta of youngest N% samples (target state)
        old_reference: mean beta of oldest N% samples (aging state)
        drift_magnitude: |young - old| per CpG
        """
        cutoff_young = np.percentile(ages, self.young_percentile)
        cutoff_old = np.percentile(ages, 100.0 - self.young_percentile)

        young_mask = ages <= cutoff_young
        old_mask = ages >= cutoff_old

        young_data = X.loc[young_mask].values.astype(np.float32)
        old_data = X.loc[old_mask].values.astype(np.float32)

        self.young_reference = np.mean(young_data, axis=0)
        self.old_reference = np.mean(old_data, axis=0)
        self.drift_magnitude = np.abs(self.old_reference - self.young_reference)
        self.feature_names = X.columns.tolist()
        self.is_built = True

        return self

    def _align_sample(self, sample: pd.Series) -> np.ndarray:
        """Align sample to reference feature space."""
        aligned = pd.Series(self.young_reference, index=self.feature_names)
        common = [c for c in sample.index if c in aligned.index]
        aligned[common] = sample[common].values
        return aligned.values.astype(np.float32)

    def simulate_intervention(
        self,
        sample_beta: np.ndarray,
        clock,
        intervention_pct: float,
        target: str = 'young'
    ) -> dict:
        """
        Apply partial reprogramming to a single sample.

        Parameters:
        - sample_beta: array of CpG beta values
        - clock: fitted BiologicalClock instance
        - intervention_pct: % of highest-drift CpGs to reset (0-100)
        - target: 'young' (move toward young reference) or 'neutral' (move to 0.5 correction)

        Returns dict with original and post-intervention biological ages.
        """
        if not self.is_built:
            raise RuntimeError("Build references first.")

        beta = sample_beta.astype(np.float32).copy()
        ref = self.young_reference.copy()

        # Select top-N% highest drift CpGs as intervention targets
        n_intervene = max(1, int(len(beta) * intervention_pct / 100.0))
        top_drift_idx = np.argsort(self.drift_magnitude)[-n_intervene:]

        # Apply intervention: move beta toward young reference
        beta_new = beta.copy()
        if target == 'young':
            beta_new[top_drift_idx] = (
                beta[top_drift_idx] * 0.0 + ref[top_drift_idx] * 1.0
            )
        elif target == 'partial':
            # 80% of the way to young reference
            beta_new[top_drift_idx] = (
                beta[top_drift_idx] * 0.2 + ref[top_drift_idx] * 0.8
            )

        # Clip to valid range
        beta_new = np.clip(beta_new, 0.0, 1.0)

        # Predict biological age before and after
        feat_names = clock.feature_names
        sample_df_orig = pd.DataFrame(
            [beta], columns=self.feature_names
        ).reindex(columns=feat_names, fill_value=0.5)
        sample_df_new = pd.DataFrame(
            [beta_new], columns=self.feature_names
        ).reindex(columns=feat_names, fill_value=0.5)

        bio_age_orig = float(clock.predict(sample_df_orig)[0])
        bio_age_new = float(clock.predict(sample_df_new)[0])
        years_reversed = bio_age_orig - bio_age_new

        return {
            'intervention_pct': intervention_pct,
            'n_cpgs_intervened': n_intervene,
            'bio_age_before': bio_age_orig,
            'bio_age_after': bio_age_new,
            'years_reversed': years_reversed,
            'pct_reversed': float(years_reversed / max(bio_age_orig - 18, 1) * 100),
            'beta_original': beta,
            'beta_reprogrammed': beta_new,
            'delta_beta': beta_new - beta,
        }

    def reversal_curve(
        self,
        sample_beta: np.ndarray,
        clock,
        steps: int = 20
    ) -> pd.DataFrame:
        """
        Generate full reversal curve: years_reversed vs intervention_pct.
        Sweeps from 1% to 100% intervention.
        """
        percentages = np.linspace(1, 100, steps)
        results = []
        for pct in percentages:
            res = self.simulate_intervention(sample_beta, clock, pct, target='young')
            results.append({
                'intervention_pct': pct,
                'bio_age_after': res['bio_age_after'],
                'years_reversed': res['years_reversed'],
                'pct_age_reversed': res['pct_reversed'],
                'n_cpgs': res['n_cpgs_intervened']
            })
        return pd.DataFrame(results)

    def batch_reversal_potential(
        self,
        X: pd.DataFrame,
        ages: pd.Series,
        clock,
        intervention_pct: float = 30.0
    ) -> pd.DataFrame:
        """
        Compute reversal potential for all samples at a given intervention level.
        """
        results = []
        feat_arr = X.values.astype(np.float32)
        for i, (idx, row) in enumerate(X.iterrows()):
            beta = feat_arr[i]
            res = self.simulate_intervention(beta, clock, intervention_pct)
            results.append({
                'sample_id': idx,
                'chronological_age': float(ages.iloc[i]),
                'bio_age_before': res['bio_age_before'],
                'bio_age_after': res['bio_age_after'],
                'years_reversed': res['years_reversed'],
                'intervention_pct': intervention_pct
            })
        return pd.DataFrame(results)

    def get_drift_landscape(self, top_n: int = 100) -> pd.DataFrame:
        """Most drifted CpGs: highest |young_ref - old_ref|."""
        if not self.is_built:
            raise RuntimeError("Build references first.")
        idx = np.argsort(self.drift_magnitude)[-top_n:][::-1]
        return pd.DataFrame({
            'cpg': [self.feature_names[i] for i in idx],
            'young_beta': self.young_reference[idx],
            'old_beta': self.old_reference[idx],
            'drift': self.drift_magnitude[idx],
            'drift_direction': np.where(
                self.old_reference[idx] > self.young_reference[idx],
                'Hypermethylated with age',
                'Hypomethylated with age'
            )
        })
