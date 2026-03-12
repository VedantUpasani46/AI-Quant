"""
Module 38: Regime Detection with Hidden Markov Model
====================================================
TRUE HMM implementation with temporal dynamics and state transitions.

Key Difference from GMM:
- HMM: Models temporal dependencies via transition probabilities
- GMM: Static mixture model with no temporal structure
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Tuple, Dict


class RegimeDetectorHMM:
    """
    Market regime detection using proper Hidden Markov Model.
    
    Unlike GMM (static mixture), HMM models:
    - State transitions over time
    - Temporal dependencies
    - Regime persistence/switching dynamics
    
    This is the correct implementation for regime detection.
    """
    
    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        """
        Initialize HMM regime detector.
        
        Args:
            n_regimes: Number of market regimes (default: 4)
            random_state: Random seed for reproducibility
        
        Raises:
            ValueError: If n_regimes < 2
        """
        if n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")
        
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.regime_labels = None
        self.is_fitted = False
    
    def fit(self, returns: pd.Series, volatility: pd.Series) -> 'RegimeDetectorHMM':
        """
        Fit HMM to market data.
        
        Args:
            returns: Daily returns series
            volatility: Daily volatility series (e.g., 20-day rolling std)
        
        Returns:
            self for method chaining
        
        Raises:
            ValueError: If input shapes don't match or contain NaN
        """
        # Input validation
        if len(returns) != len(volatility):
            raise ValueError(
                f"Length mismatch: returns={len(returns)}, volatility={len(volatility)}"
            )
        
        if returns.isna().any():
            raise ValueError("returns contains NaN values")
        
        if volatility.isna().any():
            raise ValueError("volatility contains NaN values")
        
        if len(returns) < 100:
            raise ValueError(
                f"Insufficient data: need >= 100 points, got {len(returns)}"
            )
        
        # Prepare features: returns and volatility
        X = np.column_stack([returns.values, volatility.values])
        
        # Initialize GaussianHMM
        try:
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=self.random_state,
                verbose=False
            )
            
            # Fit model
            self.model.fit(X)
            
            # Assign interpretable regime labels
            self._assign_regime_labels()
            
            self.is_fitted = True
            
        except Exception as e:
            raise RuntimeError(f"HMM fitting failed: {str(e)}")
        
        return self
    
    def _assign_regime_labels(self) -> None:
        """
        Assign interpretable labels based on emission means.
        
        Regimes:
        - Bull Low-Vol: positive return, low volatility
        - Bull High-Vol: positive return, high volatility
        - Bear High-Vol: negative return, high volatility
        - Sideways: low absolute return, moderate volatility
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        
        means = self.model.means_  # Shape: (n_regimes, 2)
        labels = []
        
        for mean_ret, mean_vol in means:
            if mean_ret > 0.001:
                if mean_vol < np.median(means[:, 1]):
                    labels.append('Bull Low-Vol')
                else:
                    labels.append('Bull High-Vol')
            elif mean_ret < -0.001:
                labels.append('Bear High-Vol')
            else:
                labels.append('Sideways')
        
        self.regime_labels = labels
    
    def predict(self, returns: pd.Series, volatility: pd.Series) -> np.ndarray:
        """
        Predict regime sequence using Viterbi algorithm.
        
        Key difference from GMM: Uses Viterbi to find most likely
        state sequence given temporal dependencies.
        
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
        
        if len(returns) != len(volatility):
            raise ValueError("Length mismatch in predict()")
        
        X = np.column_stack([returns.values, volatility.values])
        
        try:
            regime_ids = self.model.predict(X)
            return regime_ids
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_regime_probabilities(self, returns: pd.Series, 
                                 volatility: pd.Series) -> np.ndarray:
        """
        Get posterior probabilities of each regime.
        
        Returns probabilities using forward-backward algorithm.
        
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
        
        X = np.column_stack([returns.values, volatility.values])
        
        try:
            # Use forward-backward to get posteriors
            posteriors = self.model.predict_proba(X)
            return posteriors
        except Exception as e:
            raise RuntimeError(f"Probability calculation failed: {str(e)}")
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get regime transition probability matrix.
        
        This is what makes HMM different from GMM!
        Shows probability of switching from one regime to another.
        
        Returns:
            Transition matrix of shape (n_regimes, n_regimes)
            where [i, j] = P(state_t = j | state_{t-1} = i)
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before get_transition_matrix()")
        
        return self.model.transmat_
    
    def get_regime_persistence(self) -> Dict[str, float]:
        """
        Calculate how persistent each regime is.
        
        Persistence = P(stay in same regime) = diagonal of transition matrix
        
        Returns:
            Dict mapping regime label to persistence probability
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before get_regime_persistence()")
        
        persistence = {}
        trans_matrix = self.model.transmat_
        
        for i, label in enumerate(self.regime_labels):
            persistence[label] = trans_matrix[i, i]
        
        return persistence
    
    def current_regime(self, returns: pd.Series, 
                      volatility: pd.Series) -> Tuple[str, float]:
        """
        Get current regime and confidence.
        
        Args:
            returns: Recent returns
            volatility: Recent volatility
        
        Returns:
            Tuple of (regime_label, confidence)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before current_regime()")
        
        regime_id = self.predict(returns, volatility)[-1]
        probabilities = self.get_regime_probabilities(returns, volatility)[-1]
        confidence = probabilities[regime_id]
        
        return self.regime_labels[regime_id], confidence


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 38: REGIME DETECTION WITH TRUE HMM")
    print("=" * 70)
    
    # Generate synthetic market data with regime shifts
    np.random.seed(42)
    n = 500
    
    # Create data with 4 distinct regimes
    returns_list = []
    vol_list = []
    
    # Bull Low-Vol (150 days)
    returns_list.append(np.random.normal(0.0008, 0.01, 150))
    vol_list.append(np.ones(150) * 0.01)
    
    # Bear High-Vol (100 days)
    returns_list.append(np.random.normal(-0.002, 0.03, 100))
    vol_list.append(np.ones(100) * 0.03)
    
    # Sideways (150 days)
    returns_list.append(np.random.normal(0.0, 0.015, 150))
    vol_list.append(np.ones(150) * 0.015)
    
    # Bull High-Vol (100 days)
    returns_list.append(np.random.normal(0.001, 0.025, 100))
    vol_list.append(np.ones(100) * 0.025)
    
    returns = pd.Series(np.concatenate(returns_list))
    volatility = pd.Series(np.concatenate(vol_list))
    
    print(f"\nGenerated {len(returns)} days of market data")
    print(f"  Returns: mean={returns.mean():.4f}, std={returns.std():.4f}")
    print(f"  Volatility: mean={volatility.mean():.4f}")
    
    # Fit HMM
    print(f"\n── Fitting HMM with {4} regimes ──")
    detector = RegimeDetectorHMM(n_regimes=4, random_state=42)
    
    try:
        detector.fit(returns, volatility)
        print("  ✓ HMM fitted successfully")
        
        # Show detected regimes
        print(f"\n── Detected Regimes ──")
        for i, label in enumerate(detector.regime_labels):
            mean_ret, mean_vol = detector.model.means_[i]
            print(f"  Regime {i}: {label}")
            print(f"    Mean Return: {mean_ret:.4f}")
            print(f"    Mean Volatility: {mean_vol:.4f}")
        
        # Show transition matrix (KEY DIFFERENCE FROM GMM!)
        print(f"\n── Transition Matrix (HMM-specific!) ──")
        trans_matrix = detector.get_transition_matrix()
        print("  Rows = from state, Cols = to state")
        for i, label in enumerate(detector.regime_labels):
            print(f"\n  From {label}:")
            for j, to_label in enumerate(detector.regime_labels):
                prob = trans_matrix[i, j]
                print(f"    → {to_label}: {prob:.3f}")
        
        # Show regime persistence
        print(f"\n── Regime Persistence ──")
        persistence = detector.get_regime_persistence()
        for label, persist in persistence.items():
            print(f"  {label}: {persist:.3f} (P(stay in regime))")
        
        # Predict current regime
        print(f"\n── Current Regime Detection ──")
        recent_returns = returns.tail(50)
        recent_vol = volatility.tail(50)
        current, confidence = detector.current_regime(recent_returns, recent_vol)
        print(f"  Current: {current}")
        print(f"  Confidence: {confidence:.2%}")
        
        # Show full sequence
        regime_sequence = detector.predict(returns, volatility)
        print(f"\n── Regime Sequence Statistics ──")
        unique, counts = np.unique(regime_sequence, return_counts=True)
        for regime_id, count in zip(unique, counts):
            pct = count / len(regime_sequence) * 100
            print(f"  {detector.regime_labels[regime_id]}: {count} days ({pct:.1f}%)")
        
        print(f"\n✓ Module 38 complete - Proper HMM implementation!")
        print(f"\nKEY DIFFERENCE FROM GMM:")
        print(f"  - HMM models state transitions over time")
        print(f"  - GMM is static, no temporal structure")
        print(f"  - HMM uses Viterbi algorithm for prediction")
        print(f"  - HMM has transition probabilities (see above)")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise
