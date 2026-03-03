"""
=============================================================================
LIQUIDITY TRADER — OPTIMAL LIMIT ORDER PLACEMENT  (v2 — Enhanced)
=============================================================================
Improvements over v1:
  1. GARCH alpha floor  — prevents alpha collapsing to zero on short samples
  2. Overnight gap as exogenous variable in GARCH and ACD
  3. GARCH fitted on intraday bar returns (~5,850 obs) not daily (~15 obs)
  4. Volatility regime detector — separates high/low vol days

Walk-Forward Architecture:
  Phase 1 — Training    : Days  1–15  (market structure learning)
  Phase 2 — Calibration : Days 16–22  (λ tuning, model validation)
  Phase 3 — Test        : Days 23–27  (out-of-sample, last 5 days)

Models:
  A) Unconditional Empirical  — historical fill-rate + left-truncated close dist
  B) Conditional ARMA-GARCH   — intraday-fitted GARCH(1,1) + ACD AR(1) proxy
                                 overnight gap + volatility regime as exog vars

Volume weighting applied throughout. Both BUY and SELL evaluated.
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────
CSV_PATH    = 'data/ARVN_1min_RTH_40days.csv'
N_TRAIN     = 15
N_CALIB     = 7
N_TEST      = 5

LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
OFFSET_GRID = np.arange(0.002, 0.060, 0.001)
MAX_OFFSET  = 0.05
MIN_VOL     = 10          # min bar volume to count as liquid
TICK        = 0.001
ALPHA_FLOOR = 0.05        # Fix #1: prevent GARCH alpha collapsing to zero


# ─────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['date']     = df['datetime'].dt.date
    df['time_str'] = df['datetime'].dt.strftime('%H:%M')
    df['vol_weight'] = df['volume'].clip(lower=1.0)

    # Intraday returns (same-day only)
    df['prev_close'] = df.groupby('date')['close'].shift(1)
    df['ret']        = (df['close'] - df['prev_close']) / df['prev_close']
    df['intraday']   = ~df['ret'].isna()

    # Daily summary
    daily = df.groupby('date', sort=True).agg(
        day_open     = ('open',   'first'),
        day_close    = ('close',  'last'),
        day_high     = ('high',   'max'),
        day_low      = ('low',    'min'),
        total_volume = ('volume', 'sum'),
        bar_count    = ('close',  'count'),
    ).reset_index()

    # Fix #2: Overnight gap = (today_open - yesterday_close) / yesterday_close
    daily['overnight_gap'] = (
        daily['day_open'] - daily['day_close'].shift(1)
    ) / daily['day_close'].shift(1).replace(0, np.nan)
    daily['overnight_gap'] = daily['overnight_gap'].fillna(0.0)

    # Liquid high/low (bars with volume >= MIN_VOL)
    liq     = df[df['volume'] >= MIN_VOL]
    liq_lo  = liq.groupby('date')['low'].min().rename('liq_day_low')
    liq_hi  = liq.groupby('date')['high'].max().rename('liq_day_high')
    daily   = daily.merge(liq_lo, on='date', how='left')
    daily   = daily.merge(liq_hi, on='date', how='left')
    daily['liq_day_low']  = daily['liq_day_low'].fillna(daily['day_low'])
    daily['liq_day_high'] = daily['liq_day_high'].fillna(daily['day_high'])

    daily['daily_return'] = (daily['day_close'] - daily['day_open']) / daily['day_open']
    daily['daily_range']  = daily['day_high'] - daily['day_low']

    # Fix #4: Volatility regime — high vol day if daily_range > rolling median
    daily['vol_regime'] = (
        daily['daily_range'] > daily['daily_range'].rolling(5, min_periods=1).median()
    ).astype(int)  # 1 = high-vol day, 0 = low-vol day

    return df, daily


# ─────────────────────────────────────────────
# 2.  UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def compute_utility(pe, ue, une_values, une_weights, lam):
    if len(une_values) == 0:
        return -lam * pe * (1 - pe) * ue**2, 0.0, pe * (1-pe) * ue**2
    w     = np.array(une_weights, dtype=float)
    w     = w / w.sum()
    e_une = float(np.dot(w, une_values))
    v_une = float(np.dot(w, une_values**2) - e_une**2)
    e_u   = pe * ue + (1 - pe) * e_une
    v_u   = (1 - pe) * (pe * (ue - e_une)**2 + v_une)
    return float(e_u - lam * v_u), float(e_u), float(v_u)


# ─────────────────────────────────────────────
# 3.  MODEL A — UNCONDITIONAL EMPIRICAL
# ─────────────────────────────────────────────
class UnconditionalModel:
    def fit(self, daily_train):
        self.train = daily_train.copy()

    def _pe_buy(self, open_p, offset):
        limit = open_p * (1 - offset)
        w     = self.train['total_volume'].values
        filled= (self.train['liq_day_low'] <= limit).astype(float)
        return float(np.clip(np.average(filled, weights=w), 1e-6, 1-1e-6))

    def _pe_sell(self, open_p, offset):
        limit = open_p * (1 + offset)
        w     = self.train['total_volume'].values
        filled= (self.train['liq_day_high'] >= limit).astype(float)
        return float(np.clip(np.average(filled, weights=w), 1e-6, 1-1e-6))

    def _une_buy(self, open_p, offset):
        limit  = open_p * (1 - offset)
        mask   = self.train['liq_day_low'] > limit
        subset = self.train[mask]
        if len(subset) == 0:
            return np.array([0.0]), np.array([1.0])
        return (open_p - subset['day_close']).values, subset['total_volume'].values

    def _une_sell(self, open_p, offset):
        limit  = open_p * (1 + offset)
        mask   = self.train['liq_day_high'] < limit
        subset = self.train[mask]
        if len(subset) == 0:
            return np.array([0.0]), np.array([1.0])
        return (subset['day_close'] - open_p).values, subset['total_volume'].values

    def _best_offset(self, open_p, lam, side):
        best, best_off = -np.inf, OFFSET_GRID[0]
        for off in OFFSET_GRID:
            if off > MAX_OFFSET:
                break
            if side == 'buy':
                pe = self._pe_buy(open_p, off)
                ue = off * open_p
                u, w = self._une_buy(open_p, off)
            else:
                pe = self._pe_sell(open_p, off)
                ue = off * open_p
                u, w = self._une_sell(open_p, off)
            comb, _, _ = compute_utility(pe, ue, u, w, lam)
            if comb > best:
                best, best_off = comb, off
        return best_off, best

    def optimal_limit_buy(self, open_p, lam):
        return self._best_offset(open_p, lam, 'buy')

    def optimal_limit_sell(self, open_p, lam):
        return self._best_offset(open_p, lam, 'sell')

    def update(self, row):
        pass  # Unconditional model doesn't update


# ─────────────────────────────────────────────
# 4.  MODEL B — CONDITIONAL (INTRADAY GARCH + ACD + REGIME)
# ─────────────────────────────────────────────
class IntraGARCH:
    """
    Fix #3: GARCH(1,1) fitted on intraday 1-min returns (~5,850 obs vs 15 daily).
    Fix #1: Alpha floor at ALPHA_FLOOR = 0.05.
    Fix #2: Overnight gap as exogenous shock to sigma^2.
    """
    def __init__(self):
        self.mu    = 0.0
        self.omega = 1e-6
        self.alpha = ALPHA_FLOOR
        self.beta  = 0.85
        self.gamma = 0.0   # Fix #2: coefficient on overnight_gap^2
        self.fitted= False

    def _sigma2_path(self, rets, omega, alpha, beta, mu):
        n      = len(rets)
        s2     = np.full(n, max(np.var(rets), 1e-8))
        for t in range(1, n):
            e2   = (rets[t-1] - mu)**2
            s2[t]= omega + alpha * e2 + beta * s2[t-1]
            s2[t]= max(s2[t], 1e-10)
        return s2

    def _neg_ll(self, params, rets):
        mu, omega, alpha, beta = params
        # Fix #1: enforce alpha floor
        alpha = max(alpha, ALPHA_FLOOR)
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.9999:
            return 1e12
        s2 = self._sigma2_path(rets, omega, alpha, beta, mu)
        ll = -0.5 * np.sum(np.log(2*np.pi*s2) + (rets - mu)**2 / s2)
        return -ll

    def fit(self, df_bars_train):
        """Fit on intraday returns from training bars."""
        rets = df_bars_train[df_bars_train['intraday']]['ret'].dropna().values
        rets = rets[np.isfinite(rets)]
        if len(rets) < 50:
            return
        mu0  = np.mean(rets)
        v0   = np.var(rets)
        x0   = [mu0, v0*0.05, 0.1, 0.80]
        bds  = [(-0.01, 0.01), (1e-10, v0), (ALPHA_FLOOR, 0.40), (0.50, 0.9999)]
        try:
            res = minimize(self._neg_ll, x0, args=(rets,),
                           method='L-BFGS-B', bounds=bds,
                           options={'maxiter':1000,'ftol':1e-10})
            mu, omega, alpha, beta = res.x
            self.mu    = float(mu)
            self.omega = float(omega)
            self.alpha = float(max(alpha, ALPHA_FLOOR))
            self.beta  = float(beta)
        except Exception:
            self.mu = float(np.mean(rets))
        self.fitted       = True
        self._last_sigma2 = float(np.var(rets))
        self._last_ret    = float(rets[-1]) if len(rets) > 0 else 0.0

    def forecast_daily_vol(self, overnight_gap=0.0, last_ret=None, last_sigma2=None):
        """
        Aggregate intraday GARCH to a daily volatility forecast.
        Fix #2: Scale up sigma^2 if overnight_gap is large.
        """
        if not self.fitted:
            return 0.02
        lr = last_ret    if last_ret    is not None else self._last_ret
        ls = last_sigma2 if last_sigma2 is not None else self._last_sigma2
        e2      = (lr - self.mu)**2
        s2_next = self.omega + self.alpha * e2 + self.beta * ls
        # Fix #2: overnight gap amplifies next-day variance
        s2_next += 0.5 * overnight_gap**2
        s2_next  = max(s2_next, 1e-10)
        # Scale to daily: multiply by number of intraday bars
        s2_daily = s2_next * 390
        return float(np.sqrt(s2_daily))


class ACDModel:
    """AR(1) ACD with volume weighting and overnight gap (Fix #2)."""
    def __init__(self):
        self.omega    = 0.005
        self.beta     = 0.70
        self.gamma    = 0.0   # Fix #2: overnight gap effect on expected fluctuation
        self.psi_mean = 0.03
        self.fitted   = False

    def fit(self, daily_train):
        ranges  = daily_train['daily_range'].values
        gaps    = daily_train['overnight_gap'].values
        weights = daily_train['total_volume'].values.astype(float)
        weights = weights / weights.sum()
        if len(ranges) < 3:
            self.psi_mean = float(np.mean(ranges))
            return
        self.psi_mean = float(np.average(ranges, weights=weights))
        X = ranges[:-1]
        G = np.abs(gaps[1:])   # overnight gap absolute value
        Y = ranges[1:]
        W = weights[1:]
        # Weighted OLS: Y = omega + beta*X + gamma*|gap|
        A    = np.column_stack([np.ones_like(X), X, G])
        WA   = A * W[:, None]
        try:
            coef = np.linalg.lstsq(WA.T @ A, WA.T @ Y, rcond=None)[0]
            self.omega = float(np.clip(coef[0], 1e-5, 0.5))
            self.beta  = float(np.clip(coef[1], 0.0, 0.95))
            self.gamma = float(np.clip(coef[2], 0.0, 2.0))
        except Exception:
            pass
        self.fitted = True

    def forecast(self, last_range=None, overnight_gap=0.0):
        if not self.fitted:
            return self.psi_mean
        lr  = last_range if last_range is not None else self.psi_mean
        psi = self.omega + self.beta * lr + self.gamma * abs(overnight_gap)
        return float(max(psi, 1e-4))


class ConditionalModel:
    """Full conditional model: IntraGARCH + ACD + Regime detector."""

    def __init__(self):
        self.garch = IntraGARCH()
        self.acd   = ACDModel()
        self._last_range   = 0.05
        self._last_ret     = 0.0
        self._last_sigma2  = 1e-5
        self._overnight    = 0.0
        self._regime       = 0

    def fit(self, df_bars_train, daily_train):
        self.garch.fit(df_bars_train)
        self.acd.fit(daily_train)
        self._last_range  = float(daily_train['daily_range'].values[-1])
        self._last_ret    = float(daily_train['daily_return'].values[-1])
        self._last_sigma2 = float(np.var(daily_train['daily_return'].values))
        self._overnight   = float(daily_train['overnight_gap'].values[-1])
        self._regime      = int(daily_train['vol_regime'].values[-1])

    def _exec_prob(self, open_p, offset, psi, direction):
        if direction == 'buy':
            dist = open_p - open_p*(1-offset)
        else:
            dist = open_p*(1+offset) - open_p
        z  = dist / max(psi, 1e-6)
        pe = 1.0 - norm.cdf(z)
        # Fix #4: regime adjustment — widen distribution on high-vol days
        if self._regime == 1:
            z2 = dist / max(psi * 1.3, 1e-6)
            pe = max(pe, 1.0 - norm.cdf(z2))
        return float(np.clip(pe, 1e-6, 1-1e-6))

    def _une(self, open_p, sigma_daily, direction):
        """Truncated-mean UNE via Inverse Mills Ratio (Eq. 7.32 / Eq. 12)."""
        if direction == 'buy':
            mu_r  = -abs(self.garch.mu) * 390    # expected intraday drift
            z     = -mu_r / max(sigma_daily, 1e-6)
            imr   = norm.pdf(z) / max(1 - norm.cdf(z), 1e-9)
            e_ret = mu_r + sigma_daily * imr
            e_une = -open_p * e_ret
            v_une = (open_p * sigma_daily)**2 * max(1 + z*imr - imr**2, 1e-8)
        else:
            mu_r  = abs(self.garch.mu) * 390
            z     = mu_r / max(sigma_daily, 1e-6)
            imr   = norm.pdf(z) / max(1 - norm.cdf(z), 1e-9)
            e_ret = mu_r - sigma_daily * imr
            e_une = open_p * e_ret
            v_une = (open_p * sigma_daily)**2 * max(1 + z*imr - imr**2, 1e-8)
        return float(e_une), float(v_une)

    def _best_offset(self, open_p, lam, direction):
        psi      = self.acd.forecast(self._last_range, self._overnight)
        sigma_d  = self.garch.forecast_daily_vol(
                       self._overnight, self._last_ret, self._last_sigma2)
        best, best_off = -np.inf, OFFSET_GRID[0]
        for off in OFFSET_GRID:
            if off > MAX_OFFSET:
                break
            pe         = self._exec_prob(open_p, off, psi, direction)
            ue         = off * open_p
            e_une, v_u = self._une(open_p, sigma_d, direction)
            e_u        = pe * ue + (1 - pe) * e_une
            v_u2       = (1 - pe) * (pe * (ue - e_une)**2 + v_u)
            comb       = e_u - lam * v_u2
            if comb > best:
                best, best_off = comb, off
        return best_off, best

    def optimal_limit_buy(self, open_p, lam):
        return self._best_offset(open_p, lam, 'buy')

    def optimal_limit_sell(self, open_p, lam):
        return self._best_offset(open_p, lam, 'sell')

    def update(self, row):
        """Update state after observing a new day."""
        self._last_range   = float(row['daily_range'])
        self._last_ret     = float(row['daily_return'])
        self._overnight    = float(row['overnight_gap'])
        self._regime       = int(row['vol_regime'])
        e2                 = (self._last_ret - self.garch.mu)**2
        self._last_sigma2  = (self.garch.omega
                              + self.garch.alpha * e2
                              + self.garch.beta  * self._last_sigma2)


# ─────────────────────────────────────────────
# 5.  LAMBDA CALIBRATION
# ─────────────────────────────────────────────
def calibrate_lambda(model, daily_calib, side):
    best_lam, best_score = LAMBDA_GRID[0], -np.inf
    for lam in LAMBDA_GRID:
        total = 0.0
        for _, row in daily_calib.iterrows():
            op = row['day_open']
            if side == 'buy':
                off, _ = model.optimal_limit_buy(op, lam)
                limit  = op * (1 - off)
                filled = row['liq_day_low'] <= limit
                pnl    = (op - limit) if filled else (op - row['day_close'])
            else:
                off, _ = model.optimal_limit_sell(op, lam)
                limit  = op * (1 + off)
                filled = row['liq_day_high'] >= limit
                pnl    = (limit - op) if filled else (row['day_close'] - op)
            total += pnl
        if total > best_score:
            best_score, best_lam = total, lam
    return best_lam, best_score


# ─────────────────────────────────────────────
# 6.  SIMULATION + REPORTING
# ─────────────────────────────────────────────
def simulate_day(model, row, lam, side):
    op = row['day_open']
    if side == 'buy':
        off, comb = model.optimal_limit_buy(op, lam)
        limit     = op * (1 - off)
        filled    = row['liq_day_low'] <= limit
        pnl       = (op - limit) if filled else (op - row['day_close'])
    else:
        off, comb = model.optimal_limit_sell(op, lam)
        limit     = op * (1 + off)
        filled    = row['liq_day_high'] >= limit
        pnl       = (limit - op) if filled else (row['day_close'] - op)
    return {
        'date'       : str(row['date']),
        'open'       : round(float(op), 4),
        'limit'      : round(float(limit), 4),
        'offset_pct' : round(float(off)*100, 2),
        'filled'     : bool(filled),
        'outcome'    : 'FILL' if filled else 'PENALTY',
        'pnl'        : round(float(pnl), 5),
        'day_ret_pct': round(float(row['daily_return'])*100, 2),
        'overnight'  : round(float(row['overnight_gap'])*100, 2),
        'regime'     : int(row['vol_regime']),
        'volume'     : int(row['total_volume']),
        'utility'    : round(float(comb), 6),
        'lambda'     : lam,
        'side'       : side,
    }


def run_phase(model, daily_ph, lam, side, update=False):
    results = []
    for _, row in daily_ph.iterrows():
        results.append(simulate_day(model, row, lam, side))
        if update and hasattr(model, 'update'):
            model.update(row)
    return results


def summarise(results, label):
    df  = pd.DataFrame(results)
    n   = len(df)
    fills = df[df['filled']]
    pens  = df[~df['filled']]
    print(f"\n{'─'*72}")
    print(f"  {label}")
    print(f"{'─'*72}")
    print(f"  Days: {n}  |  Fills: {len(fills)}  |  Penalties: {len(pens)}  "
          f"|  Fill Rate: {len(fills)/n*100:.1f}%")
    print(f"  Total P&L  : {df['pnl'].sum():+.5f}  (vs market order at open)")
    if len(fills):
        print(f"  Fill P&L   : {fills['pnl'].sum():+.5f}  "
              f"(avg {fills['pnl'].mean():+.5f}/day)")
    if len(pens):
        print(f"  Penalty P&L: {pens['pnl'].sum():+.5f}  "
              f"(avg {pens['pnl'].mean():+.5f}/day)")
    print(f"  Avg Offset : {df['offset_pct'].mean():.2f}%   λ={df['lambda'].iloc[0]}")
    print(f"\n  {'Date':<12}{'Open':>7}{'Limit':>8}{'Off%':>6}{'Result':>9}"
          f"{'P&L':>10}{'DayRet%':>9}{'Gap%':>7}{'Vol':>8}")
    print(f"  {'─'*70}")
    for _, r in df.iterrows():
        mk = '✓' if r['filled'] else '✗'
        print(f"  {r['date']:<12}{r['open']:>7.3f}{r['limit']:>8.3f}"
              f"{r['offset_pct']:>5.1f}%{mk+r['outcome']:>9}"
              f"{r['pnl']:>+10.5f}{r['day_ret_pct']:>8.2f}%"
              f"{r['overnight']:>6.2f}%{r['volume']:>8,}")
    return df


def compare(r_unc, r_cond, label, side):
    u = pd.DataFrame(r_unc)
    c = pd.DataFrame(r_cond)
    print(f"\n{'═'*72}")
    print(f"  COMPARISON  |  {label}  |  {side.upper()}")
    print(f"{'═'*72}")
    metrics = [
        ('Total P&L vs open',  u['pnl'].sum(),              c['pnl'].sum()),
        ('Fill Rate %',        u['filled'].mean()*100,       c['filled'].mean()*100),
        ('Avg Offset %',       u['offset_pct'].mean(),       c['offset_pct'].mean()),
        ('Fill P&L',           u[u['filled']]['pnl'].sum()  if u['filled'].any() else 0,
                               c[c['filled']]['pnl'].sum()  if c['filled'].any() else 0),
        ('Penalty P&L',        u[~u['filled']]['pnl'].sum() if (~u['filled']).any() else 0,
                               c[~c['filled']]['pnl'].sum() if (~c['filled']).any() else 0),
    ]
    print(f"  {'Metric':<28}{'Unconditional':>16}{'Conditional':>16}{'Winner':>12}")
    print(f"  {'─'*70}")
    for name, uv, cv in metrics:
        w = '← UNC' if uv > cv else ('→ COND' if cv > uv else '  TIED')
        print(f"  {name:<28}{uv:>16.5f}{cv:>16.5f}{w:>12}")


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  LIQUIDITY TRADER v2  —  Enhanced GARCH + Overnight Gap + Regime")
    print(f"  Walk-Forward: {N_TRAIN} Train | {N_CALIB} Calibration | {N_TEST} Test")
    print("=" * 72)

    df_bars, daily = load_data(CSV_PATH)
    total = len(daily)
    needed = N_TRAIN + N_CALIB + N_TEST
    if total < needed:
        raise ValueError(f"Need {needed} days, only {total} available.")

    d_train = daily.iloc[:N_TRAIN].reset_index(drop=True)
    d_calib = daily.iloc[N_TRAIN:N_TRAIN+N_CALIB].reset_index(drop=True)
    d_test  = daily.iloc[N_TRAIN+N_CALIB:N_TRAIN+N_CALIB+N_TEST].reset_index(drop=True)

    # Intraday bars for training
    train_dates = set(d_train['date'].astype(str))
    df_train_bars = df_bars[df_bars['date'].astype(str).isin(train_dates)]

    print(f"\n  Phase 1 — Train : {d_train['date'].iloc[0]} → {d_train['date'].iloc[-1]}  ({len(d_train)} days, {len(df_train_bars):,} bars)")
    print(f"  Phase 2 — Calib : {d_calib['date'].iloc[0]} → {d_calib['date'].iloc[-1]}  ({len(d_calib)} days)")
    print(f"  Phase 3 — Test  : {d_test['date'].iloc[0]} → {d_test['date'].iloc[-1]}   ({len(d_test)} days)")

    # Market stats
    print(f"\n{'─'*72}")
    print(f"  TRAINING MARKET STATISTICS")
    print(f"{'─'*72}")
    print(f"  Mean daily range      : ${d_train['daily_range'].mean():.4f}  ({d_train['daily_range'].mean()/d_train['day_open'].mean()*100:.2f}%)")
    print(f"  Mean daily return     : {d_train['daily_return'].mean()*100:.3f}%   Std: {d_train['daily_return'].std()*100:.3f}%")
    print(f"  Up/Down days          : {(d_train['daily_return']>0).sum()} / {(d_train['daily_return']<0).sum()}")
    print(f"  Mean overnight gap    : {d_train['overnight_gap'].mean()*100:.3f}%   Std: {d_train['overnight_gap'].std()*100:.3f}%")
    print(f"  High-vol regime days  : {d_train['vol_regime'].sum()} / {len(d_train)}")
    print(f"  Median volume/day     : {d_train['total_volume'].median():,.0f}")
    intra_rets = df_train_bars[df_train_bars['intraday']]['ret'].dropna()
    print(f"  Intraday return std   : {intra_rets.std()*100:.4f}% per bar")
    print(f"  Intraday observations : {len(intra_rets):,}")

    all_results = {}

    for side in ['buy', 'sell']:
        print(f"\n\n{'#'*72}")
        print(f"#  SIDE: {side.upper()}")
        print(f"{'#'*72}")

        # Fit
        unc  = UnconditionalModel()
        cond = ConditionalModel()
        unc.fit(d_train)
        cond.fit(df_train_bars, d_train)

        print(f"\n  [Fitted Model Parameters — {side.upper()}]")
        print(f"  IntraGARCH : μ={cond.garch.mu:.6f}  α={cond.garch.alpha:.4f}  "
              f"β={cond.garch.beta:.4f}  ω={cond.garch.omega:.2e}")
        print(f"  ACD        : ω={cond.acd.omega:.5f}  β={cond.acd.beta:.4f}  "
              f"γ(gap)={cond.acd.gamma:.4f}  ψ̄={cond.acd.psi_mean:.4f}")
        print(f"  Alpha floor: {ALPHA_FLOOR}  (prevents α=0 collapse)")

        # Sample forecast
        psi_f   = cond.acd.forecast(cond._last_range, cond._overnight)
        sig_f   = cond.garch.forecast_daily_vol(cond._overnight)
        print(f"  Next-day ψ forecast : {psi_f:.4f}  (expected price fluctuation)")
        print(f"  Next-day σ forecast : {sig_f:.4f}  (daily volatility)")

        # Calibrate
        print(f"\n  Calibrating λ on Phase 2...")
        lam_u, sc_u  = calibrate_lambda(unc,  d_calib, side)
        lam_c, sc_c  = calibrate_lambda(cond, d_calib, side)
        print(f"  Unconditional  → best λ = {lam_u}   calib P&L = {sc_u:+.5f}")
        print(f"  Conditional    → best λ = {lam_c}   calib P&L = {sc_c:+.5f}")

        # Calibration period
        uc_res = run_phase(unc,  d_calib, lam_u, side)
        cc_res = run_phase(cond, d_calib, lam_c, side, update=True)
        summarise(uc_res, f"UNCONDITIONAL | Calibration | {side.upper()}")
        summarise(cc_res, f"CONDITIONAL   | Calibration | {side.upper()}")
        compare(uc_res, cc_res, 'CALIBRATION', side)

        # Test period
        ut_res = run_phase(unc,  d_test, lam_u, side)
        ct_res = run_phase(cond, d_test, lam_c, side, update=True)
        summarise(ut_res, f"UNCONDITIONAL | Test (OOS) | {side.upper()}")
        summarise(ct_res, f"CONDITIONAL   | Test (OOS) | {side.upper()}")
        compare(ut_res, ct_res, 'TEST (OUT-OF-SAMPLE)', side)

        all_results[side] = {
            'unc_calib': pd.DataFrame(uc_res), 'cond_calib': pd.DataFrame(cc_res),
            'unc_test' : pd.DataFrame(ut_res), 'cond_test' : pd.DataFrame(ct_res),
            'lam_unc'  : lam_u, 'lam_cond': lam_c,
            'cond_model': cond,
        }

    # Final summary
    print(f"\n\n{'═'*72}")
    print(f"  FINAL SUMMARY — TEST PERIOD (OUT-OF-SAMPLE)")
    print(f"  {'Model':<22}{'Side':<6}{'Fill%':>7}{'P&L':>11}{'Verdict'}")
    print(f"  {'─'*68}")
    for side in ['buy','sell']:
        r = all_results[side]
        for lbl, dff in [('Unconditional', r['unc_test']),
                         ('Conditional',   r['cond_test'])]:
            pnl = dff['pnl'].sum()
            fp  = dff['filled'].mean()*100
            v   = '✓ BEATS MARKET ORDER' if pnl > 0 else '✗ UNDERPERFORMS'
            print(f"  {lbl:<22}{side:<6}{fp:>6.1f}%{pnl:>10.5f}  {v}")

    print(f"\n  P&L measured vs immediate market order at open (baseline = 0).")
    print(f"  Positive P&L = limit strategy saved money vs market order.")
    print(f"{'═'*72}\n")
    return all_results


if __name__ == '__main__':
    results = main()