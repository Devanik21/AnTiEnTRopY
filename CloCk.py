"""
clock.py — Biological Age Clock Engine
Epigenetic clock via ElasticNet regression on CpG methylation beta values.
Based on Horvath (2013) methodology, trained on user data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

MAX_ITER = 2000
RANDOM_STATE = 42

# Subset of known Horvath 2013 clock CpGs (intersection detection)
HORVATH_HYPER = [  # methylation increases with age
    'cg16867657', 'cg24724428', 'cg20822990', 'cg07547549', 'cg22736354',
    'cg17861230', 'cg03607117', 'cg10501210', 'cg21801378', 'cg09809672',
    'cg04528819', 'cg07064406', 'cg22512670', 'cg12895696', 'cg21132947',
]
HORVATH_HYPO = [  # methylation decreases with age
    'cg14361627', 'cg01620164', 'cg13924996', 'cg12804289', 'cg05727880',
    'cg21878650', 'cg25557739', 'cg09809672', 'cg13777234', 'cg10480329',
]


class BiologicalClock:
    """
    Epigenetic age prediction via penalized regression.
    Trains ElasticNet on top-N variable CpGs to predict chronological age,
    then extracts biological age as the predictor (not the target).
    Age Acceleration = Biological Age - Chronological Age residual.
    """

    def __init__(self, n_variable_cpgs: int = 5000):
        self.n_variable_cpgs = n_variable_cpgs
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.metrics = {}
        self.cpg_coefs = None
        self.horvath_overlap = []

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        variances = X.var(axis=0)
        top = variances.nlargest(self.n_variable_cpgs).index
        return X[top]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BiologicalClock':
        """
        Fit the biological age clock.
        X: DataFrame of CpG beta values (samples × CpGs)
        y: Series of chronological ages
        """
        X_sub = self._select_features(X)
        self.feature_names = X_sub.columns.tolist()
        self.horvath_overlap = [c for c in self.feature_names
                                if c in HORVATH_HYPER + HORVATH_HYPO]

        X_arr = X_sub.values.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X_arr)
        y_arr = y.values.astype(np.float32)

        # ElasticNet with cross-validated alpha and l1_ratio
        self.model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            alphas=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE
        )
        self.model.fit(X_scaled, y_arr)

        y_pred = self.model.predict(X_scaled)

        # 5-fold CV MAE using the best parameters from ElasticNetCV
        best_model = ElasticNet(
            alpha=self.model.alpha_,
            l1_ratio=self.model.l1_ratio_,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE
        )
        cv_scores = cross_val_score(
            best_model, X_scaled, y_arr,
            cv=3, scoring='neg_mean_absolute_error'
        )
        self.cv_scores = -cv_scores

        self.metrics = {
            'train_mae': float(mean_absolute_error(y_arr, y_pred)),
            'train_r2': float(r2_score(y_arr, y_pred)),
            'cv_mae': float(cv_scores.mean()),
            'cv_mae_std': float(cv_scores.std()),
            'n_cpgs_total': len(self.feature_names),
            'n_cpgs_nonzero': int(np.sum(self.model.coef_ != 0)),
            'alpha': float(self.model.alpha_),
            'l1_ratio': float(self.model.l1_ratio_),
            'horvath_overlap': len(self.horvath_overlap),
        }

        # Store CpG coefficients
        self.cpg_coefs = pd.DataFrame({
            'cpg': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coef': np.abs(self.model.coef_)
        }).sort_values('abs_coef', ascending=False)
        self.cpg_coefs['direction'] = self.cpg_coefs['coefficient'].apply(
            lambda c: 'Hypermethylated' if c > 0 else 'Hypomethylated'
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Clock not fitted.")
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0.5).astype(np.float32)
        X_scaled = self.scaler.transform(X_aligned)
        return self.model.predict(X_scaled).astype(np.float32)

    def get_age_acceleration(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Intrinsic age acceleration: residual of bio_age ~ chrono_age regression.
        Removes age-dependent bias to get true epigenetic dysregulation score.
        """
        bio_age = self.predict(X)
        slope, intercept, r, p, se = stats.linregress(y.values, bio_age)
        expected = slope * y.values + intercept
        accel = bio_age - expected

        return pd.DataFrame({
            'sample_id': X.index if hasattr(X, 'index') else range(len(y)),
            'chronological_age': y.values.astype(np.float32),
            'biological_age': bio_age,
            'expected_biological_age': expected.astype(np.float32),
            'age_acceleration': accel.astype(np.float32),
            'accelerated': accel > 0
        })

    def get_top_cpgs(self, n: int = 50) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Clock not fitted.")
        active = self.cpg_coefs[self.cpg_coefs['abs_coef'] > 0]
        result = active.head(n).copy()
        result['is_horvath'] = result['cpg'].isin(HORVATH_HYPER + HORVATH_HYPO)
        return result

    def get_methylation_age_matrix(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Full biological age table for all samples."""
        accel_df = self.get_age_acceleration(X, y)
        return accel_df.sort_values('age_acceleration', ascending=False)
