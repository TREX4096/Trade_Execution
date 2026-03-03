"""
=============================================================================
LIQUIDITY TRADER — OPTIMAL LIMIT ORDER PLACEMENT
=============================================================================
Walk-Forward Architecture:
  Phase 1 — Training    : Days  1–25  (market structure learning)
  Phase 2 — Calibration : Days 26–35  (λ tuning, model validation)
  Phase 3 — Test        : Days 36–40  (out-of-sample evaluation)

Models:
  A) Unconditional Empirical  — historical fill-rate + left-truncated close dist
  B) Conditional ARMA-GARCH   — AR(1) for drift + hand-rolled GARCH(1,1) for vol
                                 ACD AR(1) proxy for execution probability

Volume weighting applied throughout.
Both BUY and SELL evaluated separately.
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────
CSV_PATH      = 'data/ARCT_1min_RTH_40days.csv'

N_TRAIN       = 15   # days for Phase 1 — market structure  (scaled: 27-day dataset)
N_CALIB       = 7    # days for Phase 2 — lambda calibration
N_TEST        = 5    # days for Phase 3 — out-of-sample test (last 5 days, unchanged)

LAMBDA_GRID   = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]  # risk aversion candidates
OFFSET_GRID   = np.arange(0.002, 0.060, 0.001)          # limit offsets to search (0.2%–6%)
MAX_OFFSET    = 0.05                                      # hard cap: 5% from open
MIN_VOL_FILL  = 10                                        # minimum volume to count a bar as "liquid"
TICK          = 0.001                                     # tick size for discrete summations


# ─────────────────────────────────────────────
# 1.  DATA LOADING & PREPARATION
# ─────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['date']     = df['datetime'].dt.date
    df['time_str'] = df['datetime'].dt.strftime('%H:%M')

    # Volume weight: each bar's contribution proportional to its volume
    # Bars with 0 volume get a small floor weight so they aren't completely ignored
    df['vol_weight'] = df['volume'].clip(lower=1.0)

    # Intraday returns (only within same day — no overnight)
    df['prev_close'] = df.groupby('date')['close'].shift(1)
    df['ret']        = (df['close'] - df['prev_close']) / df['prev_close']

    # Daily summary
    daily = df.groupby('date', sort=True).agg(
        day_open      = ('open',       'first'),
        day_close     = ('close',      'last'),
        day_high      = ('high',       'max'),
        day_low       = ('low',        'min'),
        total_volume  = ('volume',     'sum'),
        bar_count     = ('close',      'count'),
    ).reset_index()

    # Liquid day_low / day_high — only from bars with volume >= MIN_VOL_FILL
    liq = df[df['volume'] >= MIN_VOL_FILL]
    liq_low  = liq.groupby('date')['low'].min().rename('liq_day_low')
    liq_high = liq.groupby('date')['high'].max().rename('liq_day_high')
    daily    = daily.merge(liq_low,  on='date', how='left')
    daily    = daily.merge(liq_high, on='date', how='left')
    # Fall back to raw high/low if no liquid bars exist that day
    daily['liq_day_low']  = daily['liq_day_low'].fillna(daily['day_low'])
    daily['liq_day_high'] = daily['liq_day_high'].fillna(daily['day_high'])

    daily['daily_return'] = (daily['day_close'] - daily['day_open']) / daily['day_open']
    daily['daily_range']  = daily['day_high'] - daily['day_low']

    return df, daily


# ─────────────────────────────────────────────
# 2.  UTILITY FUNCTIONS  (Mean-Variance Framework)
# ─────────────────────────────────────────────
def compute_utility(pe, ue, une_values, une_weights, lam):
    """
    Equations 7.35–7.37 from thesis, discrete version.

    pe          : scalar, execution probability
    ue          : scalar, payoff if executed (= open - limit for buy)
    une_values  : array of UNE outcomes (open - close) for non-fill scenarios
    une_weights : array of probability weights for each UNE outcome
    lam         : risk aversion scalar
    """
    if len(une_values) == 0:
        # Edge case: all historical days filled — assume zero penalty
        e_une = 0.0
        v_une = 0.0
    else:
        w      = une_weights / une_weights.sum()   # normalise
        e_une  = np.dot(w, une_values)
        v_une  = np.dot(w, une_values**2) - e_une**2

    # Expected utility  (Eq 7.35 discrete)
    e_u = pe * ue + (1 - pe) * e_une

    # Variance  (Eq 7.36 discrete)
    v_u = (1 - pe) * (pe * (ue - e_une)**2 + v_une)

    # Combined objective  (Eq 7.37)
    combined = e_u - lam * v_u
    return combined, e_u, v_u


# ─────────────────────────────────────────────
# 3.  MODEL A — UNCONDITIONAL EMPIRICAL
# ─────────────────────────────────────────────
class UnconditionalModel:
    """
    Execution probability from empirical fill-rate of historical High-Low ranges.
    Non-execution price distribution from left-truncated historical closes.
    Volume-weighted throughout.
    """

    def __init__(self):
        self.daily_train = None   # stored training days

    def fit(self, daily_train):
        self.daily_train = daily_train.copy()

    def execution_prob_buy(self, open_price, offset):
        """P(day_low <= limit) weighted by total_volume of each training day."""
        limit     = open_price * (1 - offset)
        drop      = open_price - limit              # how far price must fall
        train     = self.daily_train
        # Volume-weighted fill rate
        filled    = (train['liq_day_low'] <= limit).astype(float)
        weights   = train['total_volume'].values
        pe        = np.average(filled, weights=weights)
        return float(np.clip(pe, 1e-6, 1 - 1e-6))

    def execution_prob_sell(self, open_price, offset):
        limit     = open_price * (1 + offset)
        train     = self.daily_train
        filled    = (train['liq_day_high'] >= limit).astype(float)
        weights   = train['total_volume'].values
        pe        = np.average(filled, weights=weights)
        return float(np.clip(pe, 1e-6, 1 - 1e-6))

    def une_distribution_buy(self, open_price, offset):
        """
        Left-truncated distribution of UNE for buy:
        Only include days where price did NOT fall to limit
        (adverse selection — these are the days we are forced to buy at close).
        UNE = open_price - close  (positive = price fell, we benefit; negative = price rose, we pay)
        Volume-weighted.
        """
        limit  = open_price * (1 - offset)
        train  = self.daily_train
        mask   = train['liq_day_low'] > limit          # not filled
        subset = train[mask]
        if len(subset) == 0:
            return np.array([0.0]), np.array([1.0])
        une_vals = (open_price - subset['day_close']).values
        weights  = subset['total_volume'].values
        return une_vals, weights

    def une_distribution_sell(self, open_price, offset):
        limit  = open_price * (1 + offset)
        train  = self.daily_train
        mask   = train['liq_day_high'] < limit
        subset = train[mask]
        if len(subset) == 0:
            return np.array([0.0]), np.array([1.0])
        une_vals = (subset['day_close'] - open_price).values
        weights  = subset['total_volume'].values
        return une_vals, weights

    def optimal_limit_buy(self, open_price, lam):
        best_combined, best_offset = -np.inf, OFFSET_GRID[0]
        for offset in OFFSET_GRID:
            if offset > MAX_OFFSET:
                break
            pe              = self.execution_prob_buy(open_price, offset)
            ue              = offset * open_price
            une_vals, uwts  = self.une_distribution_buy(open_price, offset)
            combined, _, _  = compute_utility(pe, ue, une_vals, uwts, lam)
            if combined > best_combined:
                best_combined = combined
                best_offset   = offset
        return best_offset, best_combined

    def optimal_limit_sell(self, open_price, lam):
        best_combined, best_offset = -np.inf, OFFSET_GRID[0]
        for offset in OFFSET_GRID:
            if offset > MAX_OFFSET:
                break
            pe              = self.execution_prob_sell(open_price, offset)
            ue              = offset * open_price
            une_vals, uwts  = self.une_distribution_sell(open_price, offset)
            combined, _, _  = compute_utility(pe, ue, une_vals, uwts, lam)
            if combined > best_combined:
                best_combined = combined
                best_offset   = offset
        return best_offset, best_combined


# ─────────────────────────────────────────────
# 4.  MODEL B — CONDITIONAL (ARMA-GARCH + ACD)
# ─────────────────────────────────────────────
class ARMAGARCHModel:
    """
    Hand-rolled AR(1)-GARCH(1,1) for the close price returns.
    Fitted via MLE with Normal innovations (Eq 7.34).
    """

    def __init__(self):
        self.phi   = 0.0    # AR(1) coefficient
        self.omega = 1e-6   # GARCH omega
        self.alpha = 0.1    # GARCH alpha (shock)
        self.beta  = 0.85   # GARCH beta  (persistence)
        self.mu    = 0.0    # mean return
        self.fitted = False

    def _garch_variance(self, rets, omega, alpha, beta, mu):
        n      = len(rets)
        sigma2 = np.full(n, np.var(rets))
        for t in range(1, n):
            e2         = (rets[t-1] - mu)**2
            sigma2[t]  = omega + alpha * e2 + beta * sigma2[t-1]
            sigma2[t]  = max(sigma2[t], 1e-10)
        return sigma2

    def _neg_loglik(self, params, rets):
        mu, phi, omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
            return 1e10
        resid  = rets[1:] - mu - phi * rets[:-1]
        sigma2 = self._garch_variance(resid, omega, alpha, beta, 0.0)
        sigma2 = np.maximum(sigma2, 1e-10)
        ll     = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + resid**2 / sigma2)
        return -ll

    def fit(self, daily_train):
        rets = daily_train['daily_return'].values
        rets = rets[np.isfinite(rets)]
        if len(rets) < 5:
            return
        mu0    = np.mean(rets)
        var0   = np.var(rets)
        x0     = [mu0, 0.0, var0 * 0.05, 0.1, 0.8]
        bounds = [(-0.2, 0.2), (-0.9, 0.9), (1e-8, 0.1), (0.0, 0.5), (0.0, 0.99)]
        try:
            res = minimize(self._neg_loglik, x0, args=(rets,),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-9})
            if res.success or res.fun < self._neg_loglik(x0, rets):
                self.mu, self.phi, self.omega, self.alpha, self.beta = res.x
        except Exception:
            self.mu = np.mean(rets)
        self.fitted = True
        self._last_sigma2 = np.var(rets)
        self._last_ret    = rets[-1] if len(rets) > 0 else 0.0

    def forecast(self, last_ret=None, last_sigma2=None):
        """
        1-step ahead conditional mean and variance.
        Returns (mu_forecast, sigma_forecast)
        """
        if not self.fitted:
            return 0.0, 0.02
        lr = last_ret    if last_ret    is not None else self._last_ret
        ls = last_sigma2 if last_sigma2 is not None else self._last_sigma2
        mu_hat    = self.mu + self.phi * lr
        e2        = (lr - self.mu)**2
        sigma2_f  = self.omega + self.alpha * e2 + self.beta * ls
        sigma2_f  = max(sigma2_f, 1e-8)
        return float(mu_hat), float(np.sqrt(sigma2_f))


class ACDModel:
    """
    AR(1) proxy for Autoregressive Conditional Duration.
    Models the expected intraday price fluctuation ψ_i.
    Volume-weighted fitting.
    """

    def __init__(self):
        self.omega     = 0.001
        self.beta      = 0.7
        self.psi_mean  = 0.03
        self.fitted    = False

    def fit(self, daily_train):
        """
        Fit AR(1) on daily High-Low ranges (our proxy for price fluctuation ψ).
        Volume-weighted so big-volume days dominate the estimate.
        """
        ranges  = daily_train['daily_range'].values
        weights = daily_train['total_volume'].values.astype(float)
        weights = weights / weights.sum()

        if len(ranges) < 3:
            self.psi_mean = np.mean(ranges)
            return

        self.psi_mean = float(np.average(ranges, weights=weights))

        # AR(1): range_t = omega + beta * range_{t-1}
        X = ranges[:-1]
        Y = ranges[1:]
        W = weights[1:]
        # Weighted least squares
        denom = np.sum(W * X**2)
        if denom > 0:
            self.beta  = float(np.clip(np.sum(W * X * Y) / denom, 0.0, 0.95))
            self.omega = float(np.clip(
                np.average(Y - self.beta * X, weights=W), 1e-5, 0.5))
        self.fitted = True

    def forecast(self, last_range=None):
        """1-step ahead expected price fluctuation ψ_i."""
        if not self.fitted:
            return self.psi_mean
        lr   = last_range if last_range is not None else self.psi_mean
        psi  = self.omega + self.beta * lr
        return float(max(psi, 1e-4))


class ConditionalModel:
    """
    Wraps ACDModel (execution prob) + ARMAGARCHModel (non-execution risk).
    """

    def __init__(self):
        self.acd   = ACDModel()
        self.garch = ARMAGARCHModel()

    def fit(self, daily_train):
        self.acd.fit(daily_train)
        self.garch.fit(daily_train)
        self._last_range   = daily_train['daily_range'].values[-1]
        self._last_ret     = daily_train['daily_return'].values[-1]
        self._last_sigma2  = np.var(daily_train['daily_return'].values)

    def _exec_prob_buy(self, open_price, offset, psi):
        """
        P_E = 1 - F_ε( (open - limit) / psi )
        Using Normal CDF as ε distribution (Eq 7.23).
        Volume-weighted via psi which was volume-weighted in ACD fit.
        """
        limit    = open_price * (1 - offset)
        drop_req = open_price - limit
        z        = drop_req / max(psi, 1e-6)
        pe       = 1.0 - norm.cdf(z)
        return float(np.clip(pe, 1e-6, 1 - 1e-6))

    def _exec_prob_sell(self, open_price, offset, psi):
        limit    = open_price * (1 + offset)
        rise_req = limit - open_price
        z        = rise_req / max(psi, 1e-6)
        pe       = 1.0 - norm.cdf(z)
        return float(np.clip(pe, 1e-6, 1 - 1e-6))

    def _une_buy(self, open_price, mu_ret, sigma_ret):
        """
        UNE for buy: open - close  when order not filled.
        Expected close = open * (1 + mu_ret).
        Truncated mean via Inverse Mills Ratio (Eq 12 from summary).
        UNE positive = we gain (price fell); negative = we lose (price rose).
        """
        # E[close | close > limit effectively, i.e. no fill]
        # Simplified: use normal approximation with GARCH sigma
        # UNE = open - close → E[UNE] = -open * mu_ret  (approx, ignoring truncation)
        # With truncation (price stayed above limit throughout):
        # E[UNE] ≈ open * (-mu_ret - sigma_ret * phi(z)/(1-Phi(z)))
        z       = -mu_ret / max(sigma_ret, 1e-6)
        imr     = norm.pdf(z) / max(1 - norm.cdf(z), 1e-6)   # Inverse Mills Ratio
        e_ret   = mu_ret + sigma_ret * imr                     # truncated mean return
        e_une   = -open_price * e_ret
        v_ret   = sigma_ret**2 * (1 + z * imr - imr**2)
        v_une   = (open_price**2) * max(v_ret, 1e-8)
        return float(e_une), float(v_une)

    def _une_sell(self, open_price, mu_ret, sigma_ret):
        z       = mu_ret / max(sigma_ret, 1e-6)
        imr     = norm.pdf(z) / max(1 - norm.cdf(z), 1e-6)
        e_ret   = mu_ret - sigma_ret * imr
        e_une   = open_price * e_ret
        v_ret   = sigma_ret**2 * (1 + z * imr - imr**2)
        v_une   = (open_price**2) * max(v_ret, 1e-8)
        return float(e_une), float(v_une)

    def optimal_limit_buy(self, open_price, lam):
        psi          = self.acd.forecast(self._last_range)
        mu_r, sig_r  = self.garch.forecast(self._last_ret, self._last_sigma2)
        best_combined, best_offset = -np.inf, OFFSET_GRID[0]

        for offset in OFFSET_GRID:
            if offset > MAX_OFFSET:
                break
            pe            = self._exec_prob_buy(open_price, offset, psi)
            ue            = offset * open_price
            e_une, v_une  = self._une_buy(open_price, mu_r, sig_r)
            # Scalar version of compute_utility
            e_u           = pe * ue + (1 - pe) * e_une
            v_u           = (1 - pe) * (pe * (ue - e_une)**2 + v_une)
            combined      = e_u - lam * v_u
            if combined > best_combined:
                best_combined = combined
                best_offset   = offset

        return best_offset, best_combined

    def optimal_limit_sell(self, open_price, lam):
        psi          = self.acd.forecast(self._last_range)
        mu_r, sig_r  = self.garch.forecast(self._last_ret, self._last_sigma2)
        best_combined, best_offset = -np.inf, OFFSET_GRID[0]

        for offset in OFFSET_GRID:
            if offset > MAX_OFFSET:
                break
            pe            = self._exec_prob_sell(open_price, offset, psi)
            ue            = offset * open_price
            e_une, v_une  = self._une_sell(open_price, mu_r, sig_r)
            e_u           = pe * ue + (1 - pe) * e_une
            v_u           = (1 - pe) * (pe * (ue - e_une)**2 + v_une)
            combined      = e_u - lam * v_u
            if combined > best_combined:
                best_combined = combined
                best_offset   = offset

        return best_offset, best_combined

    def update(self, day_row):
        """Update last observed range/return after each day."""
        self._last_range  = day_row['daily_range']
        self._last_ret    = day_row['daily_return']
        self._last_sigma2 = (self.garch.omega
                             + self.garch.alpha * day_row['daily_return']**2
                             + self.garch.beta  * self._last_sigma2)


# ─────────────────────────────────────────────
# 5.  LAMBDA CALIBRATION  (Phase 2)
# ─────────────────────────────────────────────
def calibrate_lambda(model, daily_calib, side='buy'):
    """
    Grid-search λ on calibration days.
    Score = total simulated P&L across calibration days.
    """
    best_lam, best_score = LAMBDA_GRID[0], -np.inf

    for lam in LAMBDA_GRID:
        total_pnl = 0.0
        for _, row in daily_calib.iterrows():
            open_p = row['day_open']
            if side == 'buy':
                offset, _ = model.optimal_limit_buy(open_p, lam)
                limit      = open_p * (1 - offset)
                filled     = row['liq_day_low'] <= limit
                if filled:
                    pnl = open_p - limit          # saved vs open
                else:
                    pnl = open_p - row['day_close']  # forced buy at close
            else:
                offset, _ = model.optimal_limit_sell(open_p, lam)
                limit      = open_p * (1 + offset)
                filled     = row['liq_day_high'] >= limit
                if filled:
                    pnl = limit - open_p
                else:
                    pnl = row['day_close'] - open_p

            total_pnl += pnl

        if total_pnl > best_score:
            best_score = total_pnl
            best_lam   = lam

    return best_lam, best_score


# ─────────────────────────────────────────────
# 6.  SIMULATION ENGINE
# ─────────────────────────────────────────────
def simulate_day(model, row, lam, side='buy'):
    """
    Simulate one trading day.
    Returns a dict with all relevant metrics.
    """
    open_p = row['day_open']

    if side == 'buy':
        offset, combined = model.optimal_limit_buy(open_p, lam)
        limit            = open_p * (1 - offset)
        filled           = row['liq_day_low'] <= limit
        if filled:
            pnl    = open_p - limit     # savings vs market order at open
            pnl_vs_close = row['day_close'] - limit   # vs end-of-day
        else:
            pnl    = open_p - row['day_close']         # forced to buy at close
            pnl_vs_close = 0.0
    else:
        offset, combined = model.optimal_limit_sell(open_p, lam)
        limit            = open_p * (1 + offset)
        filled           = row['liq_day_high'] >= limit
        if filled:
            pnl    = limit - open_p
            pnl_vs_close = limit - row['day_close']
        else:
            pnl    = row['day_close'] - open_p
            pnl_vs_close = 0.0

    # Baseline: immediate market order at open = 0 savings
    return {
        'date'            : str(row['date']),
        'open'            : round(open_p, 4),
        'limit'           : round(limit,  4),
        'offset_pct'      : round(offset * 100, 2),
        'filled'          : bool(filled),
        'outcome'         : 'FILL' if filled else 'PENALTY',
        'pnl_vs_open'     : round(pnl, 5),          # vs immediate market order
        'pnl_vs_close'    : round(pnl_vs_close, 5), # vs end-of-day execution
        'day_range'       : round(row['daily_range'], 4),
        'day_return_pct'  : round(row['daily_return'] * 100, 2),
        'day_volume'      : int(row['total_volume']),
        'combined_utility': round(combined, 6),
        'lambda'          : lam,
        'side'            : side,
    }


def run_phase(model, daily_phase, lam, side, phase_name, update_model=False):
    results = []
    for _, row in daily_phase.iterrows():
        res = simulate_day(model, row, lam, side)
        res['phase'] = phase_name
        results.append(res)
        if update_model and hasattr(model, 'update'):
            model.update(row)
    return results


# ─────────────────────────────────────────────
# 7.  REPORTING
# ─────────────────────────────────────────────
def summarise(results, label):
    df  = pd.DataFrame(results)
    n   = len(df)
    fills    = df[df['filled']]
    pens     = df[~df['filled']]
    total_pnl = df['pnl_vs_open'].sum()
    fill_pnl  = fills['pnl_vs_open'].sum()
    pen_pnl   = pens['pnl_vs_open'].sum()
    fill_rate = len(fills) / n if n > 0 else 0

    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  Days          : {n}  |  Fills: {len(fills)}  |  Penalties: {len(pens)}")
    print(f"  Fill Rate     : {fill_rate*100:.1f}%")
    print(f"  Total P&L     : {total_pnl:+.5f}  (vs immediate market order at open)")
    print(f"  Fill P&L      : {fill_pnl:+.5f}")
    print(f"  Penalty P&L   : {pen_pnl:+.5f}")
    if len(fills) > 0:
        print(f"  Avg Fill Save : {fills['pnl_vs_open'].mean():+.5f} per day")
    if len(pens) > 0:
        print(f"  Avg Penalty   : {pens['pnl_vs_open'].mean():+.5f} per day")
    print(f"  Avg Offset    : {df['offset_pct'].mean():.2f}%")
    print(f"  Lambda Used   : {df['lambda'].iloc[0]}")

    # Day-by-day breakdown
    print(f"\n  {'Date':<12} {'Open':>7} {'Limit':>7} {'Off%':>5} {'Outcome':>8} "
          f"{'P&L':>9} {'DayRet%':>8} {'Volume':>8}")
    print(f"  {'─'*68}")
    for _, r in df.iterrows():
        flag = '✓' if r['filled'] else '✗'
        pnl_s = f"{r['pnl_vs_open']:+.5f}"
        print(f"  {r['date']:<12} {r['open']:>7.4f} {r['limit']:>7.4f} "
              f"{r['offset_pct']:>4.1f}% {flag+r['outcome']:>8} "
              f"{pnl_s:>9} {r['day_return_pct']:>7.2f}% {r['day_volume']:>8,}")
    return df


def compare_models(unc_results, cond_results, phase_label, side):
    u = pd.DataFrame(unc_results)
    c = pd.DataFrame(cond_results)
    print(f"\n{'═'*70}")
    print(f"  MODEL COMPARISON  |  {phase_label}  |  {side.upper()}")
    print(f"{'═'*70}")
    print(f"  {'Metric':<30} {'Unconditional':>15} {'Conditional':>15}")
    print(f"  {'─'*60}")
    metrics = [
        ('Total P&L vs Open',      u['pnl_vs_open'].sum(),    c['pnl_vs_open'].sum()),
        ('Fill Rate %',             u['filled'].mean()*100,    c['filled'].mean()*100),
        ('Avg Offset %',            u['offset_pct'].mean(),    c['offset_pct'].mean()),
        ('Fill P&L',                u[u['filled']]['pnl_vs_open'].sum() if u['filled'].any() else 0,
                                    c[c['filled']]['pnl_vs_open'].sum() if c['filled'].any() else 0),
        ('Penalty P&L',             u[~u['filled']]['pnl_vs_open'].sum() if (~u['filled']).any() else 0,
                                    c[~c['filled']]['pnl_vs_open'].sum() if (~c['filled']).any() else 0),
    ]
    for name, uv, cv in metrics:
        winner = '← BETTER' if uv > cv else ('→ BETTER' if cv > uv else '   EQUAL')
        print(f"  {name:<30} {uv:>14.5f}  {cv:>14.5f}  {winner}")


# ─────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  LIQUIDITY TRADER — OPTIMAL LIMIT ORDER PLACEMENT")
    print("  Walk-Forward: 25 Train | 10 Calibration | 5 Test")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    df_bars, daily = load_data(CSV_PATH)
    total_days = len(daily)
    print(f"\n  Loaded {total_days} trading days, {len(df_bars):,} bars")
    print(f"  Date range: {daily['date'].iloc[0]}  →  {daily['date'].iloc[-1]}")

    needed = N_TRAIN + N_CALIB + N_TEST
    if total_days < needed:
        raise ValueError(
            f"Need {needed} days, only {total_days} available. "
            f"Adjust N_TRAIN/N_CALIB/N_TEST constants at top of file.")

    # ── Split ──────────────────────────────────────────────────────────
    daily_train = daily.iloc[:N_TRAIN].reset_index(drop=True)
    daily_calib = daily.iloc[N_TRAIN:N_TRAIN+N_CALIB].reset_index(drop=True)
    daily_test  = daily.iloc[N_TRAIN+N_CALIB:].reset_index(drop=True)

    print(f"\n  Phase 1 — Train      : {daily_train['date'].iloc[0]} → {daily_train['date'].iloc[-1]}  ({len(daily_train)} days)")
    print(f"  Phase 2 — Calibration: {daily_calib['date'].iloc[0]} → {daily_calib['date'].iloc[-1]}  ({len(daily_calib)} days)")
    print(f"  Phase 3 — Test       : {daily_test['date'].iloc[0]}  → {daily_test['date'].iloc[-1]}  ({len(daily_test)} days)")

    # ── Basic market stats from training data ──────────────────────────
    print(f"\n{'─'*70}")
    print(f"  MARKET STATISTICS  (Training Period)")
    print(f"{'─'*70}")
    print(f"  Mean daily range       : ${daily_train['daily_range'].mean():.4f}  ({daily_train['daily_range'].mean()/daily_train['day_open'].mean()*100:.2f}%)")
    print(f"  Std  daily range       : ${daily_train['daily_range'].std():.4f}")
    print(f"  Mean daily return      : {daily_train['daily_return'].mean()*100:.3f}%")
    print(f"  Std  daily return      : {daily_train['daily_return'].std()*100:.3f}%")
    print(f"  Up days / Down days    : {(daily_train['daily_return']>0).sum()} / {(daily_train['daily_return']<0).sum()}")
    print(f"  Median total volume/day: {daily_train['total_volume'].median():,.0f}")

    # ── Empirical fill-rate table ──────────────────────────────────────
    print(f"\n  Empirical Fill Rate by Offset (Training, volume-weighted):")
    print(f"  {'Offset':>7} | {'Buy Fill%':>10} | {'Sell Fill%':>11}")
    print(f"  {'─'*35}")
    for off in [0.01, 0.02, 0.03, 0.04, 0.05]:
        limit_b  = daily_train['day_open'] * (1 - off)
        limit_s  = daily_train['day_open'] * (1 + off)
        wts      = daily_train['total_volume'].values
        peb      = np.average((daily_train['liq_day_low'] <= limit_b).astype(float), weights=wts)
        pes      = np.average((daily_train['liq_day_high'] >= limit_s).astype(float), weights=wts)
        print(f"  {off*100:>6.1f}% | {peb*100:>9.1f}% | {pes*100:>10.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # RUN BOTH MODELS, BOTH SIDES
    # ═══════════════════════════════════════════════════════════════════
    all_results = {}

    for side in ['buy', 'sell']:
        print(f"\n\n{'#'*70}")
        print(f"#  SIDE: {side.upper()}")
        print(f"{'#'*70}")

        # ── Fit models on training data ────────────────────────────────
        unc_model  = UnconditionalModel()
        cond_model = ConditionalModel()
        unc_model.fit(daily_train)
        cond_model.fit(daily_train)

        print(f"\n  [Conditional Model Params — {side.upper()}]")
        print(f"  GARCH  : μ={cond_model.garch.mu:.5f}  φ={cond_model.garch.phi:.4f}  "
              f"ω={cond_model.garch.omega:.2e}  α={cond_model.garch.alpha:.4f}  β={cond_model.garch.beta:.4f}")
        print(f"  ACD    : ω={cond_model.acd.omega:.5f}  β={cond_model.acd.beta:.4f}  "
              f"ψ̄={cond_model.acd.psi_mean:.4f}")

        # ── Calibrate λ on Phase 2 ─────────────────────────────────────
        print(f"\n  Calibrating λ on Phase 2 ({len(daily_calib)} days)...")
        lam_unc,  score_unc  = calibrate_lambda(unc_model,  daily_calib, side)
        lam_cond, score_cond = calibrate_lambda(cond_model, daily_calib, side)
        print(f"  Unconditional  → best λ = {lam_unc}   (calib P&L = {score_unc:+.5f})")
        print(f"  Conditional    → best λ = {lam_cond}  (calib P&L = {score_cond:+.5f})")

        # ── Phase 2 simulation (calibration period, using best λ) ──────
        unc_calib_res  = run_phase(unc_model,  daily_calib, lam_unc,  side, 'CALIB')
        cond_calib_res = run_phase(cond_model, daily_calib, lam_cond, side, 'CALIB',
                                   update_model=True)

        print(f"\n  ── CALIBRATION PERIOD RESULTS ({side.upper()}) ──")
        df_uc = summarise(unc_calib_res,  f"UNCONDITIONAL | Calibration | {side.upper()}")
        df_cc = summarise(cond_calib_res, f"CONDITIONAL   | Calibration | {side.upper()}")
        compare_models(unc_calib_res, cond_calib_res, 'CALIBRATION', side)

        # ── Phase 3 — TEST (out-of-sample) ─────────────────────────────
        unc_test_res  = run_phase(unc_model,  daily_test, lam_unc,  side, 'TEST')
        cond_test_res = run_phase(cond_model, daily_test, lam_cond, side, 'TEST',
                                  update_model=True)

        print(f"\n  ── TEST PERIOD RESULTS ({side.upper()}) ──")
        df_ut = summarise(unc_test_res,  f"UNCONDITIONAL | Test (OOS) | {side.upper()}")
        df_ct = summarise(cond_test_res, f"CONDITIONAL   | Test (OOS) | {side.upper()}")
        compare_models(unc_test_res, cond_test_res, 'TEST (OUT-OF-SAMPLE)', side)

        all_results[side] = {
            'unc_calib' : pd.DataFrame(unc_calib_res),
            'cond_calib': pd.DataFrame(cond_calib_res),
            'unc_test'  : pd.DataFrame(unc_test_res),
            'cond_test' : pd.DataFrame(cond_test_res),
            'lam_unc'   : lam_unc,
            'lam_cond'  : lam_cond,
        }

    # ═══════════════════════════════════════════════════════════════════
    # FINAL COMBINED SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'═'*70}")
    print(f"  FINAL SUMMARY — TEST PERIOD (OUT-OF-SAMPLE)")
    print(f"  {'Model':<22} {'Side':<6} {'Fill%':>7} {'P&L':>10} {'Verdict'}")
    print(f"  {'─'*66}")

    for side in ['buy', 'sell']:
        r = all_results[side]
        for label, df_r in [('Unconditional', r['unc_test']),
                             ('Conditional',   r['cond_test'])]:
            pnl      = df_r['pnl_vs_open'].sum()
            fill_pct = df_r['filled'].mean() * 100
            verdict  = '✓ BEATS MARKET ORDER' if pnl > 0 else '✗ UNDERPERFORMS'
            print(f"  {label:<22} {side:<6} {fill_pct:>6.1f}% {pnl:>10.5f}  {verdict}")

    print(f"\n  Note: P&L is measured relative to the baseline of executing")
    print(f"  immediately at the open via market order (= 0 by definition).")
    print(f"  Positive P&L = limit order strategy saved money vs market order.")
    print(f"  Negative P&L = strategy cost more than just crossing the spread.")
    print(f"\n{'═'*70}\n")

    return all_results


if __name__ == '__main__':
    results = main()