"""
Module 38B: Regime Detection with Gaussian Mixture Model
========================================================
GMM-based regime detection

"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Tuple, Dict, Optional
import warnings


class RegimeDetectorGMM:
    """
    Market regime detection using Gaussian Mixture Model.
    GMM is faster and simpler, but doesn't capture regime persistence.
    """
    
    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        """
        Initialize GMM regime detector.
        
        Args:
            n_regimes: Number of market regimes (default: 4)
            random_state: Random seed for reproducibility
        
        Raises:
            ValueError: If n_regimes < 2
        """
        if n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")
        
        if n_regimes > 10:
            warnings.warn(
                f"n_regimes={n_regimes} is large. May overfit. Typical: 3-5.",
                UserWarning
            )
        
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.regime_labels = None
        self.is_fitted = False
        self.feature_means = None
        self.feature_stds = None
    
    def fit(self, returns: pd.Series, volatility: pd.Series) -> 'RegimeDetectorGMM':
        """
        Fit GMM to market data.
        
        Args:
            returns: Daily returns series
            volatility: Daily volatility series
        
        Returns:
            self for method chaining
        
        Raises:
            ValueError: If input validation fails
            RuntimeError: If GMM fitting fails
        """
        # Input validation
        if not isinstance(returns, pd.Series):
            raise TypeError(f"returns must be pd.Series, got {type(returns)}")
        
        if not isinstance(volatility, pd.Series):
            raise TypeError(f"volatility must be pd.Series, got {type(volatility)}")
        
        if len(returns) != len(volatility):
            raise ValueError(
                f"Length mismatch: returns={len(returns)}, volatility={len(volatility)}"
            )
        
        if returns.isna().any():
            n_nan = returns.isna().sum()
            raise ValueError(f"returns contains {n_nan} NaN values")
        
        if volatility.isna().any():
            n_nan = volatility.isna().sum()
            raise ValueError(f"volatility contains {n_nan} NaN values")
        
        if len(returns) < 100:
            raise ValueError(
                f"Insufficient data: need >= 100 points, got {len(returns)}"
            )
        
        if (volatility < 0).any():
            raise ValueError("volatility contains negative values")
        
        # Prepare features
        X = np.column_stack([returns.values, volatility.values])
        
        # Store feature statistics for validation
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0)
        
        # Check for degenerate data
        if self.feature_stds[0] < 1e-6:
            raise ValueError("returns has near-zero variance")
        
        if self.feature_stds[1] < 1e-6:
            raise ValueError("volatility has near-zero variance")
        
        # Fit GMM
        try:
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=self.random_state,
                max_iter=200,
                tol=1e-4
            )
            
            self.model.fit(X)
            
            # Check convergence
            if not self.model.converged_:
                warnings.warn(
                    "GMM did not converge. Results may be unreliable.",
                    UserWarning
                )
            
            # Assign interpretable regime labels
            self._assign_regime_labels()
            
            self.is_fitted = True
            
        except Exception as e:
            raise RuntimeError(f"GMM fitting failed: {str(e)}")
        
        return self
    
    def _assign_regime_labels(self) -> None:
        """
        Assign interpretable labels based on cluster means.
        
        Regimes defined by return and volatility characteristics.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        
        means = self.model.means_  # Shape: (n_regimes, 2)
        
        if means.shape[0] != self.n_regimes:
            raise RuntimeError(
                f"Expected {self.n_regimes} components, got {means.shape[0]}"
            )
        
        labels = []
        
        # Get median volatility for reference
        median_vol = np.median(means[:, 1])
        
        for mean_ret, mean_vol in means:
            if mean_ret > 0.001:
                if mean_vol < median_vol:
                    labels.append('Bull Low-Vol')
                else:
                    labels.append('Bull High-Vol')
            elif mean_ret < -0.001:
                labels.append('Bear High-Vol')
            else:
                labels.append('Sideways')
        
        self.regime_labels = labels
    
    def predict(self, returns: pd.Series, 
                volatility: pd.Series) -> np.ndarray:
        """
        Predict regime for each time period.
        
        Note: GMM treats each observation independently (no temporal structure).
        For temporal modeling, use HMM instead.
        
        Args:
            returns: Returns series
            volatility: Volatility series
        
        Returns:
            Array of regime IDs
        
        Raises:
            RuntimeError: If model not fitted
            ValueError: If input validation fails
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        
        if len(returns) == 0:
            raise ValueError("returns is empty")
        
        if len(returns) != len(volatility):
            raise ValueError(
                f"Length mismatch in predict(): "
                f"returns={len(returns)}, volatility={len(volatility)}"
            )
        
        # Check for distribution shift
        X = np.column_stack([returns.values, volatility.values])
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        
        # Warn if data distribution has shifted significantly
        mean_shift = np.abs(X_mean - self.feature_means) / self.feature_stds
        if (mean_shift > 3).any():
            warnings.warn(
                "Input data distribution has shifted significantly from training data. "
                "Predictions may be unreliable.",
                UserWarning
            )
        
        try:
            regime_ids = self.model.predict(X)
            
            # Validate output
            if not ((regime_ids >= 0) & (regime_ids < self.n_regimes)).all():
                raise RuntimeError("Prediction produced invalid regime IDs")
            
            return regime_ids
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_regime_probabilities(self, returns: pd.Series,
                                 volatility: pd.Series) -> np.ndarray:
        """
        Get probability of each regime for each observation.
        
        Args:
            returns: Returns series
            volatility: Volatility series
        
        Returns:
            Array of shape (n_samples, n_regimes) with probabilities
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before get_regime_probabilities()")
        
        if len(returns) != len(volatility):
            raise ValueError("Length mismatch")
        
        X = np.column_stack([returns.values, volatility.values])
        
        try:
            probs = self.model.predict_proba(X)
            
            # Validate probabilities
            if not np.allclose(probs.sum(axis=1), 1.0):
                raise RuntimeError("Probabilities don't sum to 1")
            
            if (probs < 0).any() or (probs > 1).any():
                raise RuntimeError("Invalid probability values")
            
            return probs
            
        except Exception as e:
            raise RuntimeError(f"Probability calculation failed: {str(e)}")
    
    def current_regime(self, returns: pd.Series,
                      volatility: pd.Series) -> Tuple[str, float]:
        """
        Get most recent regime and confidence.
        
        Args:
            returns: Recent returns
            volatility: Recent volatility
        
        Returns:
            Tuple of (regime_label, confidence)
        
        Raises:
            RuntimeError: If model not fitted
            ValueError: If inputs empty
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before current_regime()")
        
        if len(returns) == 0:
            raise ValueError("returns is empty")
        
        try:
            regime_id = self.predict(returns, volatility)[-1]
            probabilities = self.get_regime_probabilities(returns, volatility)[-1]
            confidence = probabilities[regime_id]
            
            return self.regime_labels[regime_id], confidence
            
        except Exception as e:
            raise RuntimeError(f"current_regime() failed: {str(e)}")
    
    def get_regime_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each regime.
        
        Returns:
            Dict mapping regime label to statistics
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before get_regime_statistics()")
        
        stats = {}
        
        for i, label in enumerate(self.regime_labels):
            mean_ret, mean_vol = self.model.means_[i]
            
            # Get covariance
            cov = self.model.covariances_[i]
            std_ret = np.sqrt(cov[0, 0])
            std_vol = np.sqrt(cov[1, 1])
            
            stats[label] = {
                'mean_return': float(mean_ret),
                'mean_volatility': float(mean_vol),
                'std_return': float(std_ret),
                'std_volatility': float(std_vol),
                'weight': float(self.model.weights_[i])
            }
        
        return stats


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 38B: REGIME DETECTION WITH GMM (CORRECTLY LABELED)")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Create data with 4 regimes
    returns_list = [
        np.random.normal(0.0008, 0.01, 150),   # Bull Low-Vol
        np.random.normal(-0.002, 0.03, 100),   # Bear High-Vol
        np.random.normal(0.0, 0.015, 150),     # Sideways
        np.random.normal(0.001, 0.025, 100)    # Bull High-Vol
    ]
    
    vol_list = [
        np.ones(150) * 0.01,
        np.ones(100) * 0.03,
        np.ones(150) * 0.015,
        np.ones(100) * 0.025
    ]
    
    returns = pd.Series(np.concatenate(returns_list))
    volatility = pd.Series(np.concatenate(vol_list))
    
    print(f"\nGenerated {len(returns)} days of market data")
    
    # Fit GMM
    print(f"\n── Fitting GMM ──")
    detector = RegimeDetectorGMM(n_regimes=4, random_state=42)
    
    try:
        detector.fit(returns, volatility)
        print("  ✓ GMM fitted successfully")
        
        # Show regime statistics
        print(f"\n── Regime Statistics ──")
        stats = detector.get_regime_statistics()
        for label, stat in stats.items():
            print(f"\n  {label}:")
            print(f"    Mean Return: {stat['mean_return']:.4f}")
            print(f"    Mean Volatility: {stat['mean_volatility']:.4f}")
            print(f"    Weight: {stat['weight']:.2%}")
        
        # Current regime
        print(f"\n── Current Regime ──")
        recent_returns = returns.tail(50)
        recent_vol = volatility.tail(50)
        current, confidence = detector.current_regime(recent_returns, recent_vol)
        print(f"  Current: {current}")
        print(f"  Confidence: {confidence:.2%}")
        
        print(f"\n✓ Module 38B complete - GMM!")
        print(f"\nKEY POINT:")
        print(f"  This is GMM, NOT HMM!")
        print(f"  - GMM: Static mixture (no temporal dynamics)")
        print(f"  - HMM: Models state transitions")
        print(f"  - For temporal modeling, use HMM (module38_regime_detection_HMM.py)")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise
