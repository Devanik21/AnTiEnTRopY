"""
immortality.py — Immortality Engine
Computes epigenetic escape velocity and longevity trajectories.

Core concept (after Aubrey de Grey, formalized here epigenetically):
- Aging rate A: rate of epigenetic entropy increase per year (dH/dt_aging)
- Reversal rate R(p): entropy decrease per intervention at level p%
- Escape velocity: minimum p* such that R(p*) ≥ A per intervention cycle

If interventions are applied every T years at level p%:
  Net entropy change = A·T - R(p)
  Immortality condition: R(p) / T ≥ A  →  p ≥ p*(T)

Monte Carlo Longevity Trajectories model stochastic aging with interventions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')


class ImmortalityEngine:
    """
    Models the mathematical conditions for biological age reversal
    to outpace chronological aging — the escape velocity problem.
    """

    def __init__(self):
        self.aging_rate = None        # dEntropy/dt (per year)
        self.entropy_at_birth = None  # baseline entropy
        self.reversal_curve_data = None  # from ReversalSimulator
        self.is_calibrated = False

    def calibrate_aging_rate(self, entropy_df: pd.DataFrame) -> 'ImmortalityEngine':
        """
        Fit aging rate from population entropy data.
        entropy_df must have columns: 'chronological_age', 'mean_entropy'
        """
        ages = entropy_df['chronological_age'].values.astype(np.float64)
        entropy = entropy_df['mean_entropy'].values.astype(np.float64)

        slope, intercept, r, p, se = linregress(ages, entropy)

        self.aging_rate = float(slope)       # entropy units per year
        self.entropy_at_birth = float(intercept)
        self.calibration = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r ** 2),
            'p_value': float(p),
            'entropy_per_decade': float(slope * 10),
            'entropy_per_year': float(slope),
        }
        self.is_calibrated = True
        return self

    def set_reversal_curve(self, reversal_df: pd.DataFrame) -> 'ImmortalityEngine':
        """
        Load reversal curve data (output of ReversalSimulator.reversal_curve).
        Columns: intervention_pct, years_reversed
        """
        self.reversal_curve_data = reversal_df.copy()
        # Convert years_reversed to entropy units using aging rate
        if self.is_calibrated:
            self.reversal_curve_data['entropy_reversed'] = (
                self.reversal_curve_data['years_reversed'] * self.aging_rate
            )
        return self

    def _reversal_at_pct(self, pct: float) -> float:
        """Interpolate years reversed at a given intervention percentage."""
        if self.reversal_curve_data is None:
            return 0.0
        df = self.reversal_curve_data.sort_values('intervention_pct')
        return float(np.interp(pct, df['intervention_pct'], df['years_reversed']))

    def compute_escape_velocity(
        self,
        intervention_interval_years: float = 1.0
    ) -> dict:
        """
        Find minimum intervention level for net-zero epigenetic aging.

        Per cycle of T years:
          Aging accumulation: A · T years of biological aging
          Reversal: R(p) years reversed by intervention

        Escape condition: R(p*) = A · T
        → p* = minimum intervention % for zero net aging
        """
        if not self.is_calibrated:
            raise RuntimeError("Calibrate aging rate first.")
        if self.reversal_curve_data is None:
            raise RuntimeError("Set reversal curve first.")

        # Aging per interval in years of biological age
        aging_per_interval = self.aging_rate * intervention_interval_years / self.aging_rate
        # Above simplifies: we need years_reversed ≥ intervention_interval_years

        pct_range = self.reversal_curve_data['intervention_pct'].values
        years_range = self.reversal_curve_data['years_reversed'].values

        # Find p* where years_reversed = intervention_interval_years
        target = intervention_interval_years

        # Check if achievable at all
        max_reversible = float(np.max(years_range))
        if max_reversible < target:
            return {
                'escape_achievable': False,
                'max_reversible_years': max_reversible,
                'target_years': target,
                'intervention_interval': intervention_interval_years,
                'message': f'Maximum reversal ({max_reversible:.1f}y) < aging per interval ({target:.1f}y). Increase frequency or intensity.'
            }

        # Interpolate to find escape velocity percentage
        escape_pct = float(np.interp(target, years_range, pct_range))

        # Safety buffer: 120% intervention for longevity surplus
        surplus_pct = float(np.interp(target * 1.2, years_range, pct_range))

        return {
            'escape_achievable': True,
            'escape_velocity_pct': escape_pct,
            'surplus_pct': min(surplus_pct, 100.0),
            'max_reversible_years': max_reversible,
            'target_years': target,
            'intervention_interval': intervention_interval_years,
            'net_aging_at_escape': 0.0,
            'message': f'Escape velocity: {escape_pct:.1f}% CpG reset every {intervention_interval_years:.0f} year(s)',
        }

    def longevity_trajectory(
        self,
        initial_bio_age: float,
        initial_chrono_age: float,
        intervention_pct: float,
        intervention_interval: float,
        years_ahead: int = 50,
        n_monte_carlo: int = 500
    ) -> pd.DataFrame:
        """
        Monte Carlo simulation of biological age trajectory with interventions.
        Models stochastic aging noise around the deterministic aging rate.
        """
        if not self.is_calibrated:
            raise RuntimeError("Calibrate aging rate first.")

        reversal_at_pct = self._reversal_at_pct(intervention_pct)
        age_noise_std = 2.0  # stochastic aging noise (years, from literature)

        trajectories = []
        time_points = np.arange(0, years_ahead + 1, 0.5)

        for trial in range(n_monte_carlo):
            bio_age = initial_bio_age
            traj = []
            next_intervention = intervention_interval

            for t in time_points:
                # Stochastic aging: deterministic rate + noise
                noise = np.random.normal(0, age_noise_std * 0.1)
                annual_aging = 1.0 + noise  # ~1 biological year per chronological year

                if t > 0:
                    bio_age += annual_aging * 0.5  # 0.5 year steps

                    # Apply intervention if scheduled
                    if t >= next_intervention:
                        reversal_noise = np.random.normal(reversal_at_pct, reversal_at_pct * 0.1)
                        bio_age -= max(0, reversal_noise)
                        bio_age = max(bio_age, 18.0)  # floor at minimum adult age
                        next_intervention += intervention_interval

                traj.append(bio_age)

            trajectories.append(traj)

        traj_array = np.array(trajectories)  # (n_monte_carlo, n_timepoints)

        # Statistics across trajectories
        return pd.DataFrame({
            'years_elapsed': time_points,
            'chrono_age': initial_chrono_age + time_points,
            'bio_age_mean': traj_array.mean(axis=0),
            'bio_age_median': np.median(traj_array, axis=0),
            'bio_age_p5': np.percentile(traj_array, 5, axis=0),
            'bio_age_p25': np.percentile(traj_array, 25, axis=0),
            'bio_age_p75': np.percentile(traj_array, 75, axis=0),
            'bio_age_p95': np.percentile(traj_array, 95, axis=0),
            'no_intervention': initial_bio_age + time_points,
        })

    def compute_intervention_landscape(
        self,
        initial_bio_age: float,
        initial_chrono_age: float,
        years_ahead: int = 30
    ) -> pd.DataFrame:
        """
        Sweep intervention_pct × intervention_interval → net bio age change.
        Produces 2D landscape for contour plotting.
        """
        pct_range = np.linspace(5, 100, 20)
        interval_range = np.arange(1, 11)

        records = []
        for pct in pct_range:
            for interval in interval_range:
                reversal = self._reversal_at_pct(pct)
                # Interventions over years_ahead
                n_interventions = years_ahead / interval
                total_reversed = reversal * n_interventions
                total_aged = years_ahead  # chronological aging
                net_bio_age_change = total_aged - total_reversed
                final_bio_age = initial_bio_age + net_bio_age_change

                ev = self.compute_escape_velocity(float(interval))
                escapes = ev.get('escape_achievable', False) and pct >= ev.get('escape_velocity_pct', 101)

                records.append({
                    'intervention_pct': pct,
                    'interval_years': interval,
                    'net_bio_age_change': net_bio_age_change,
                    'final_bio_age': max(18.0, final_bio_age),
                    'total_years_reversed': total_reversed,
                    'escape_velocity': escapes,
                })
        return pd.DataFrame(records)

    def lifespan_extension_potential(
        self,
        current_bio_age: float,
        current_chrono_age: float,
        max_bio_age: float = 120.0
    ) -> dict:
        """
        At current aging rate, compute lifespan extension from escape velocity.
        """
        years_left_natural = max(0, max_bio_age - current_bio_age)
        chrono_years_left = years_left_natural  # 1:1 in natural aging

        # At escape velocity (10% surplus reversal)
        if self.reversal_curve_data is not None:
            max_rev = float(self.reversal_curve_data['years_reversed'].max())
            # If max reversal > 1 year per year: theoretically unlimited
            unlimited = max_rev >= 1.0
        else:
            unlimited = False

        return {
            'current_biological_age': current_bio_age,
            'current_chronological_age': current_chrono_age,
            'biological_age_acceleration': current_bio_age - current_chrono_age,
            'years_remaining_natural': chrono_years_left,
            'escape_velocity_achievable': unlimited,
            'max_single_reversal_years': float(
                self.reversal_curve_data['years_reversed'].max()
            ) if self.reversal_curve_data is not None else 0.0,
        }
