"""
Hidden Markov Model for Market Regime Detection
================================================
Target: Regime prediction accuracy 70%+ | Transition lag < 5 days

This module implements a full Gaussian HMM with the Baum-Welch (EM) algorithm
for unsupervised market regime detection.

Mathematical Foundation:
------------------------
HMM Components:
  π_i  = P(state_0 = i)                  [initial probabilities]
  A_ij = P(state_t = j | state_{t-1} = i) [transition matrix]
  B_i(x)= N(x; μ_i, Σ_i)                [Gaussian emission per state]

Forward variable:  α_t(i) = P(o_1..o_t, s_t=i | λ)
Backward variable: β_t(i) = P(o_{t+1}..o_T | s_t=i, λ)

Baum-Welch E-step:
  γ_t(i)  = α_t(i)β_t(i) / Σ_j α_t(j)β_t(j)
  ξ_t(i,j)= α_t(i)A_ij B_j(o_{t+1}) β_{t+1}(j) / Σ_{i,j}(same)

Baum-Welch M-step:
  π_i^new  = γ_1(i)
  A_ij^new = Σ_{t=1}^{T-1} ξ_t(i,j) / Σ_{t=1}^{T-1} γ_t(i)
  μ_i^new  = Σ_t γ_t(i) o_t / Σ_t γ_t(i)
  Σ_i^new  = Σ_t γ_t(i)(o_t - μ_i)(o_t - μ_i)^T / Σ_t γ_t(i)

Viterbi decoding (most likely state sequence):
  δ_t(i) = max_{s_1..s_{t-1}} P(s_1..s_{t-1}, s_t=i, o_1..o_t | λ)
  ψ_t(i) = argmax_j [δ_{t-1}(j) A_ji]

References:
  - Rabiner (1989). A Tutorial on Hidden Markov Models. IEEE.
  - Hamilton (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. Econometrica.
  - Ang & Timmermann (2012). Regime Changes and Financial Markets. Annual Review.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, multivariate_normal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional: hmmlearn for production validation
try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core HMM with Full Baum-Welch EM
# ---------------------------------------------------------------------------

class GaussianHMM_BaumWelch:
    """
    Gaussian Hidden Markov Model trained via Baum-Welch (EM).

    Each state emits observations from a multivariate Gaussian:
      B_i(o) = N(o; μ_i, Σ_i)

    This is a from-scratch implementation. For production use,
    hmmlearn.hmm.GaussianHMM is recommended (same math, faster C backend).
    """

    def __init__(self, n_states: int = 3, covariance_type: str = 'full',
                 n_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.is_fitted = False

        # Model parameters (set during fit)
        self.initial_probs: Optional[np.ndarray] = None   # π, shape (K,)
        self.transition_matrix: Optional[np.ndarray] = None  # A, shape (K,K)
        self.emission_params: List[Dict] = []              # [{mean, cov}, ...]
        self.log_likelihood_history: List[float] = []

    # ------------------------------------------------------------------
    # Emission probability helpers
    # ------------------------------------------------------------------

    def _log_emission_probs(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute log P(o_t | state=i) for all t and i.

        Returns:
            log_B: shape (T, K)  — log Gaussian density
        """
        T = len(observations)
        log_B = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            mu = self.emission_params[k]['mean']
            cov = self.emission_params[k]['cov']
            try:
                rv = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
                log_B[:, k] = rv.logpdf(observations)
            except Exception:
                # Fallback: diagonal approximation
                diff = observations - mu
                var = np.diag(cov).clip(1e-8)
                log_B[:, k] = -0.5 * np.sum((diff ** 2) / var, axis=1)
        return log_B

    # ------------------------------------------------------------------
    # Forward algorithm (log-scale for numerical stability)
    # ------------------------------------------------------------------

    def _forward(self, log_B: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm in log space.

        Args:
            log_B: (T, K) log emission probabilities

        Returns:
            log_alpha: (T, K)
            log_likelihood: scalar
        """
        T, K = log_B.shape
        log_alpha = np.full((T, K), -np.inf)

        # t=0
        log_alpha[0] = np.log(self.initial_probs + 1e-300) + log_B[0]

        # t=1..T-1
        log_A = np.log(self.transition_matrix + 1e-300)
        for t in range(1, T):
            # log_alpha[t, j] = log Σ_i exp(log_alpha[t-1,i] + log_A[i,j])  + log_B[t,j]
            prev = log_alpha[t - 1][:, np.newaxis] + log_A  # (K, K)
            log_alpha[t] = self._logsumexp(prev, axis=0) + log_B[t]

        log_likelihood = self._logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    # ------------------------------------------------------------------
    # Backward algorithm (log-scale)
    # ------------------------------------------------------------------

    def _backward(self, log_B: np.ndarray) -> np.ndarray:
        """
        Backward algorithm in log space.

        Returns:
            log_beta: (T, K)
        """
        T, K = log_B.shape
        log_beta = np.full((T, K), -np.inf)
        log_beta[-1] = 0.0  # log(1)

        log_A = np.log(self.transition_matrix + 1e-300)
        for t in range(T - 2, -1, -1):
            # log_beta[t, i] = log Σ_j exp(log_A[i,j] + log_B[t+1,j] + log_beta[t+1,j])
            vals = log_A + log_B[t + 1][np.newaxis, :] + log_beta[t + 1][np.newaxis, :]  # (K, K)
            log_beta[t] = self._logsumexp(vals, axis=1)

        return log_beta

    # ------------------------------------------------------------------
    # Logsumexp helper
    # ------------------------------------------------------------------

    @staticmethod
    def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Numerically stable log-sum-exp."""
        a_max = np.max(a, axis=axis, keepdims=True)
        out = np.log(np.sum(np.exp(a - a_max), axis=axis)) + np.squeeze(a_max, axis=axis)
        return out

    # ------------------------------------------------------------------
    # E-step: compute γ and ξ
    # ------------------------------------------------------------------

    def _e_step(self, observations: np.ndarray, log_B: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        E-step: compute posterior state probabilities.

        Returns:
            gamma:  (T, K)   — P(state_t=i | obs, λ)
            xi:     (T-1, K, K) — P(state_t=i, state_{t+1}=j | obs, λ)
            log_likelihood: scalar
        """
        T, K = log_B.shape
        log_alpha, log_likelihood = self._forward(log_B)
        log_beta = self._backward(log_B)
        log_A = np.log(self.transition_matrix + 1e-300)

        # γ_t(i) = α_t(i) β_t(i) / P(O|λ)
        log_gamma = log_alpha + log_beta
        log_gamma -= self._logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)  # (T, K)

        # ξ_t(i,j) for t=0..T-2
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            # log ξ_t(i,j) = log_alpha[t,i] + log_A[i,j] + log_B[t+1,j] + log_beta[t+1,j]
            log_xi_t = (log_alpha[t][:, np.newaxis]
                        + log_A
                        + log_B[t + 1][np.newaxis, :]
                        + log_beta[t + 1][np.newaxis, :])
            log_xi_t -= self._logsumexp(log_xi_t.ravel())
            xi[t] = np.exp(log_xi_t)

        return gamma, xi, log_likelihood

    # ------------------------------------------------------------------
    # M-step: re-estimate parameters
    # ------------------------------------------------------------------

    def _m_step(self, observations: np.ndarray,
                gamma: np.ndarray, xi: np.ndarray):
        """
        M-step: update π, A, μ_k, Σ_k using soft counts.
        """
        T, D = observations.shape
        K = self.n_states

        # Update π
        self.initial_probs = gamma[0] / (gamma[0].sum() + 1e-300)

        # Update A
        A_num = xi.sum(axis=0)  # (K, K)
        self.transition_matrix = A_num / (A_num.sum(axis=1, keepdims=True) + 1e-300)

        # Update emission parameters
        for k in range(K):
            gamma_k = gamma[:, k]  # (T,)
            denom = gamma_k.sum() + 1e-300

            # Mean
            mu_k = (gamma_k[:, np.newaxis] * observations).sum(axis=0) / denom

            # Covariance
            diff = observations - mu_k  # (T, D)
            cov_k = (gamma_k[:, np.newaxis, np.newaxis]
                     * diff[:, :, np.newaxis]
                     * diff[:, np.newaxis, :]).sum(axis=0) / denom
            # Regularise
            cov_k += np.eye(D) * 1e-6

            self.emission_params[k] = {'mean': mu_k, 'cov': cov_k}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self, observations: np.ndarray):
        """K-means warm start for emission parameters."""
        T, D = observations.shape
        np.random.seed(self.random_state)

        kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state,
                        n_init=10)
        labels = kmeans.fit_predict(observations)

        self.emission_params = []
        for k in range(self.n_states):
            mask = labels == k
            obs_k = observations[mask] if mask.sum() > 1 else observations
            mu = obs_k.mean(axis=0)
            cov = np.cov(obs_k.T) if obs_k.shape[0] > 1 else np.eye(D)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            cov = np.atleast_2d(cov) + np.eye(D) * 1e-4
            self.emission_params.append({'mean': mu, 'cov': cov})

        # Transition: high self-persistence
        self.transition_matrix = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.transition_matrix, 0.7)
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)

        self.initial_probs = np.ones(self.n_states) / self.n_states

    # ------------------------------------------------------------------
    # Public API: fit
    # ------------------------------------------------------------------

    def fit(self, observations: np.ndarray, n_iter: Optional[int] = None):
        """
        Fit HMM using Baum-Welch (EM) algorithm.

        Args:
            observations: (T, D) array of observations
            n_iter: override default iteration count
        """
        T, D = observations.shape
        n_iter = n_iter or self.n_iter

        print(f"\n  Training HMM (Baum-Welch EM)...")
        print(f"    States: {self.n_states}")
        print(f"    Observations: {T} timesteps × {D} features")
        print(f"    Max iterations: {n_iter}  |  Tolerance: {self.tol}")

        self._initialise(observations)
        self.log_likelihood_history = []

        prev_ll = -np.inf
        for iteration in range(n_iter):
            # ---- E-step ----
            log_B = self._log_emission_probs(observations)
            gamma, xi, log_likelihood = self._e_step(observations, log_B)

            self.log_likelihood_history.append(log_likelihood)

            # ---- Convergence check ----
            delta = log_likelihood - prev_ll
            if iteration > 0 and abs(delta) < self.tol:
                print(f"    Converged at iteration {iteration + 1}  "
                      f"(ΔlogL={delta:.6f})")
                break
            prev_ll = log_likelihood

            if (iteration + 1) % 10 == 0:
                print(f"    Iter {iteration + 1:3d} | logL = {log_likelihood:.4f}")

            # ---- M-step ----
            self._m_step(observations, gamma, xi)

        self.is_fitted = True
        print(f"\n    Final log-likelihood: {log_likelihood:.4f}")
        print(f"\n    Transition Matrix (A_ij = P(state_j | state_i)):")
        for i in range(self.n_states):
            row = "  ".join(f"{v:.3f}" for v in self.transition_matrix[i])
            print(f"      State {i + 1}: [{row}]")

    # ------------------------------------------------------------------
    # Public API: predict (Viterbi)
    # ------------------------------------------------------------------

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Decode most likely state sequence via Viterbi algorithm.

        Returns:
            states: (T,) array of state indices in {0, …, K-1}
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        T = len(observations)
        K = self.n_states

        log_B = self._log_emission_probs(observations)      # (T, K)
        log_A = np.log(self.transition_matrix + 1e-300)     # (K, K)

        # Initialise Viterbi
        delta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=int)

        delta[0] = np.log(self.initial_probs + 1e-300) + log_B[0]

        for t in range(1, T):
            # δ_t(j) = max_i [δ_{t-1}(i) + log A_ij] + log B_j(o_t)
            scores = delta[t - 1][:, np.newaxis] + log_A  # (K, K)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_B[t]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    # ------------------------------------------------------------------
    # Public API: predict_proba (smooth posteriors)
    # ------------------------------------------------------------------

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """
        Return smoothed state probabilities γ_t(i) = P(state_t=i | all obs).

        Returns:
            gamma: (T, K)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")

        log_B = self._log_emission_probs(observations)
        gamma, _, _ = self._e_step(observations, log_B)
        return gamma

    def score(self, observations: np.ndarray) -> float:
        """Return log P(observations | model) — useful for model selection."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before score().")
        log_B = self._log_emission_probs(observations)
        _, log_likelihood = self._forward(log_B)
        return log_likelihood


# ---------------------------------------------------------------------------
# Market Regime Feature Engineering
# ---------------------------------------------------------------------------

def build_regime_features(prices: pd.Series,
                           window_vol: int = 21,
                           window_mom: int = 10) -> pd.DataFrame:
    """
    Construct multi-dimensional observations for regime detection.

    Features:
        - Realised volatility (21-day)
        - Momentum (10-day return)
        - Autocorrelation of daily returns (21-day rolling)
        - Normalised volume (if available, else skipped)
    """
    ret = prices.pct_change().dropna()

    vol = ret.rolling(window_vol).std() * np.sqrt(252)
    mom = prices.pct_change(window_mom)
    autocorr = ret.rolling(window_vol).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else 0.0,
        raw=False
    )

    df = pd.DataFrame({
        'vol': vol,
        'momentum': mom,
        'autocorr': autocorr,
    }).dropna()

    return df


# ---------------------------------------------------------------------------
# Regime Labeller (human-readable names)
# ---------------------------------------------------------------------------

def label_regimes(hmm: GaussianHMM_BaumWelch,
                  regime_names: Optional[List[str]] = None) -> Dict[int, str]:
    """
    Auto-assign regime names by sorting states on volatility (emission mean[0]).

    Convention: state with lowest vol → 'Bull / Low-Vol'
                state with highest vol → 'Bear / High-Vol'
    """
    if not hmm.is_fitted:
        raise RuntimeError("Fit HMM first.")

    vols = [p['mean'][0] for p in hmm.emission_params]
    order = np.argsort(vols)  # ascending volatility

    default_names = ['Bull/Low-Vol', 'Transition', 'Bear/High-Vol',
                     'Crisis', 'Recovery']

    if regime_names is None:
        regime_names = default_names

    labels: Dict[int, str] = {}
    for rank, state_idx in enumerate(order):
        labels[int(state_idx)] = regime_names[min(rank, len(regime_names) - 1)]

    return labels


# ---------------------------------------------------------------------------
# Walk-Forward Backtest of Regime Model
# ---------------------------------------------------------------------------

@dataclass
class RegimeBacktestResult:
    states: np.ndarray
    state_labels: Dict[int, str]
    regime_returns: Dict[str, pd.Series]
    transition_matrix: np.ndarray
    log_likelihood: float
    regime_stats: pd.DataFrame


def backtest_regime_model(prices: pd.Series,
                          n_states: int = 3,
                          n_iter: int = 100,
                          train_frac: float = 0.7) -> RegimeBacktestResult:
    """
    Train HMM on first `train_frac` of data, decode full history,
    and compute per-regime return statistics.
    """
    features_df = build_regime_features(prices)
    scaler = StandardScaler()
    obs = scaler.fit_transform(features_df.values)

    split = int(len(obs) * train_frac)
    train_obs = obs[:split]

    # Fit
    model = GaussianHMM_BaumWelch(n_states=n_states, n_iter=n_iter)
    model.fit(train_obs)

    # Decode full history
    all_states = model.predict(obs)
    labels = label_regimes(model)

    # Align returns with decoded states
    returns = prices.pct_change().reindex(features_df.index).dropna()
    regime_returns: Dict[str, pd.Series] = {}
    rows = []
    for state_id, label in labels.items():
        mask = all_states == state_id
        # align mask to returns index
        mask_series = pd.Series(mask, index=features_df.index)
        aligned_mask = mask_series.reindex(returns.index).fillna(False)
        r = returns[aligned_mask.values]
        regime_returns[label] = r
        if len(r) > 0:
            rows.append({
                'Regime': label,
                'State': state_id,
                'Count': len(r),
                'Mean Ann. Return': r.mean() * 252,
                'Ann. Volatility': r.std() * np.sqrt(252),
                'Sharpe': (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
                'Max Drawdown': _max_drawdown(r),
                'Emission Vol': model.emission_params[state_id]['mean'][0],
            })

    stats_df = pd.DataFrame(rows).set_index('Regime') if rows else pd.DataFrame()

    return RegimeBacktestResult(
        states=all_states,
        state_labels=labels,
        regime_returns=regime_returns,
        transition_matrix=model.transition_matrix,
        log_likelihood=model.log_likelihood_history[-1] if model.log_likelihood_history else float('nan'),
        regime_stats=stats_df,
    )


def _max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


# ---------------------------------------------------------------------------
# IC Measurement: do regime labels add alpha?
# ---------------------------------------------------------------------------

def measure_regime_ic(prices: pd.Series, n_states: int = 3,
                      forward_period: int = 5) -> Dict[str, float]:
    """
    Measure information coefficient between regime probability and
    forward returns, to assess regime signal quality.

    Returns:
        {'ic': float, 'ic_t_stat': float, 'hit_rate': float}
    """
    features_df = build_regime_features(prices)
    scaler = StandardScaler()
    obs = scaler.fit_transform(features_df.values)

    model = GaussianHMM_BaumWelch(n_states=n_states, n_iter=100)
    model.fit(obs)

    proba = model.predict_proba(obs)  # (T, K)
    # Use lowest-vol state (bull) probability as the alpha signal
    labels = label_regimes(model)
    bull_state = [k for k, v in labels.items() if 'Bull' in v][0]
    signal = proba[:, bull_state]

    # Forward returns aligned to features index
    fwd = prices.pct_change(forward_period).shift(-forward_period)
    fwd = fwd.reindex(features_df.index)

    valid = ~np.isnan(fwd.values)
    if valid.sum() < 20:
        return {'ic': float('nan'), 'ic_t_stat': float('nan'), 'hit_rate': float('nan')}

    ic, _ = spearmanr(signal[valid], fwd.values[valid])
    n = valid.sum()
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic ** 2, 1e-10))
    hit_rate = float(np.mean(np.sign(signal[valid] - 0.5) == np.sign(fwd.values[valid])))

    return {'ic': float(ic), 'ic_t_stat': float(t_stat), 'hit_rate': hit_rate}


# ---------------------------------------------------------------------------
# Model Selection: BIC over n_states
# ---------------------------------------------------------------------------

def select_n_states(observations: np.ndarray,
                    candidates: List[int] = None,
                    n_iter: int = 50) -> Dict[str, object]:
    """
    Select optimal number of HMM states via Bayesian Information Criterion.

    BIC = -2 * logL + k * log(T)
    where k = number of free parameters.
    """
    if candidates is None:
        candidates = [2, 3, 4, 5]

    T, D = observations.shape
    results = []

    for K in candidates:
        model = GaussianHMM_BaumWelch(n_states=K, n_iter=n_iter)
        model.fit(observations)
        ll = model.score(observations)

        # Free parameters: (K-1) initial probs + K(K-1) transitions
        #                  + K*D means + K*D*(D+1)/2 covariance entries
        k_params = ((K - 1)
                    + K * (K - 1)
                    + K * D
                    + K * D * (D + 1) // 2)
        bic = -2 * ll + k_params * np.log(T)
        aic = -2 * ll + 2 * k_params

        results.append({'K': K, 'logL': ll, 'BIC': bic, 'AIC': aic,
                        'n_params': k_params, 'model': model})
        print(f"  K={K}: logL={ll:.2f}  BIC={bic:.2f}  AIC={aic:.2f}")

    best = min(results, key=lambda x: x['BIC'])
    print(f"\n  Best K by BIC: {best['K']}")
    return best


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("HMM Market Regime Detection — Full Baum-Welch EM")
    print("=" * 70)

    np.random.seed(42)

    # Simulate a regime-switching price series (3 regimes)
    T = 500
    true_states = np.zeros(T, dtype=int)
    regime_vol = [0.008, 0.015, 0.030]   # bull, transition, bear
    regime_drift = [0.0005, 0.0001, -0.0005]

    state = 0
    trans = np.array([[0.97, 0.02, 0.01],
                      [0.05, 0.90, 0.05],
                      [0.03, 0.07, 0.90]])
    returns = []
    for t in range(T):
        true_states[t] = state
        r = regime_drift[state] + regime_vol[state] * np.random.randn()
        returns.append(r)
        state = np.random.choice(3, p=trans[state])

    prices = pd.Series(100 * np.exp(np.cumsum(returns)),
                       index=pd.date_range('2018-01-01', periods=T, freq='B'))

    # Build and fit HMM
    result = backtest_regime_model(prices, n_states=3, n_iter=80)

    print("\n  Per-Regime Statistics:")
    print(result.regime_stats.to_string())

    print("\n  State Labels:", result.regime_stats.index.tolist())
    print(f"\n  Final Log-Likelihood: {result.log_likelihood:.4f}")

    # IC test
    ic_result = measure_regime_ic(prices, n_states=3, forward_period=5)
    print(f"\n  Regime Signal IC (5-day forward): {ic_result['ic']:.4f}  "
          f"(t={ic_result['ic_t_stat']:.2f})")
    print(f"  Hit Rate: {ic_result['hit_rate']:.2%}")

    # Model selection
    print("\n  Model Selection (BIC):")
    features_df = build_regime_features(prices)
    scaler = StandardScaler()
    obs = scaler.fit_transform(features_df.values)
    best = select_n_states(obs, candidates=[2, 3, 4], n_iter=40)
    print(f"  Optimal states: {best['K']}")
