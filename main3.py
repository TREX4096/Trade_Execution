import argparse
import os
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm, gengamma
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# 1. DATA PREPARATION
# =====================================================================
def prepare_data(df, price_col='close', high_col='high', low_col='low'):
    """
    Computes returns and price fluctuations as defined in the paper.
    For a buy problem, fluctuation M_T^B = sup{p_0 - p_t}
    """
    df = df.rename(columns=str.lower)
    price_col = price_col.lower()
    high_col = high_col.lower()
    low_col = low_col.lower()

    missing = [c for c in (price_col, high_col, low_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    df['return'] = df[price_col].diff()
    # Fluctuation for a buy order: how far down the price dipped from the open/previous close
    df['fluctuation'] = df[price_col].shift(1) - df[low_col]
    df['fluctuation'] = df['fluctuation'].clip(lower=1e-6)
    return df.dropna()

# =====================================================================
# 2. ACD MODEL (Execution Probability)
# =====================================================================
def acd_log_likelihood(params, fluctuations):
    """
    Computes the negative log-likelihood for a standard ACD(1,1) model 
    assuming Generalized Gamma distributed errors, mirroring Eq 7.27.
    """
    omega, alpha, beta, kappa, lambda_ = params
    n = len(fluctuations)
    psi = np.zeros(n)
    psi[0] = np.mean(fluctuations)
    
    # ACD(1,1) Conditional Expectation
    for i in range(1, n):
        psi[i] = omega + alpha * fluctuations[i-1] + beta * psi[i-1]
    
    # Prevent division by zero or negative expectations
    psi = np.clip(psi, 1e-6, None)
    
    # White noise epsilon = delta / psi
    epsilon = fluctuations / psi
    
    # Generalized Gamma Log-Likelihood (Eq 7.26 & 7.27)
    # Using scipy.stats.gengamma.logpdf
    c = kappa  # shape parameter 1
    a = lambda_ # shape parameter 2
    ll = np.sum(gengamma.logpdf(epsilon, a, c)) - np.sum(np.log(psi))
    
    # Return negative log-likelihood for minimization
    return -ll

def fit_acd_model(fluctuations):
    """Fits the ACD model using Maximum Likelihood Estimation."""
    # Initial guesses: [omega, alpha, beta, kappa, lambda]
    initial_guess = [0.01, 0.1, 0.8, 1.0, 1.0]
    bounds = [(1e-6, None), (1e-6, 1), (1e-6, 1), (0.1, 10), (0.1, 10)]
    
    # Constraint: alpha + beta < 1 for stationarity
    constraints = ({'type': 'ineq', 'fun': lambda x: 1 - (x[1] + x[2])})
    
    result = minimize(acd_log_likelihood, initial_guess, args=(fluctuations,), 
                      bounds=bounds, constraints=constraints, method='SLSQP')
    return result.x

def predict_execution_prob(acd_params, last_fluctuation, last_psi, current_price, limit_price, tick_size=0.05):
    """
    Calculates P_E(p^L) using the Generalized Gamma CDF (Eq 7.23).
    """
    omega, alpha, beta, kappa, lambda_ = acd_params
    
    # Forecast next expected fluctuation
    next_psi = omega + alpha * last_fluctuation + beta * last_psi
    
    # Threshold for execution
    threshold = (current_price - limit_price) / next_psi
    
    # P_E = 1 - F_GG(threshold)
    P_E = 1 - gengamma.cdf(threshold, a=lambda_, c=kappa)
    return P_E, next_psi

# =====================================================================
# 3. ARMA-GARCH MODEL (Non-Execution Price Density)
# =====================================================================
def fit_arma_garch(returns):
    """Fits ARMA(1,1)-GARCH(1,1) (Eq 7.28 - 7.34)."""
    model = arch_model(returns, mean='ARX', lags=1, vol='GARCH', p=1, q=1, dist='normal')
    fitted_model = model.fit(disp='off')
    return fitted_model

def get_truncated_density(fitted_garch, current_price, limit_price, eval_price):
    """
    Computes the left-truncated normal density for the closing price 
    given no execution (Eq 7.32).
    """
    forecasts = fitted_garch.forecast(horizon=1)
    mu_return = forecasts.mean.iloc[-1].values[0]
    var_return = forecasts.variance.iloc[-1].values[0]
    sigma = np.sqrt(var_return)
    
    expected_close = current_price + mu_return
    
    # Standardize
    z_eval = (eval_price - expected_close) / sigma
    z_limit = (limit_price - expected_close) / sigma
    
    # Truncation denominator: 1 - Phi(z_limit)
    prob_above_limit = 1 - norm.cdf(z_limit)
    if prob_above_limit <= 0:
        return 0.0
    
    # Truncated PDF
    pdf_eval = norm.pdf(z_eval) / sigma
    trunc_pdf = pdf_eval / prob_above_limit
    
    return trunc_pdf

# =====================================================================
# 4. UTILITY OPTIMIZATION (T-ACD-ARMA Model)
# =====================================================================
def evaluate_t_acd_arma(acd_params, fitted_garch, last_fluct, last_psi, current_price, limit_price, risk_aversion, tick_size=0.05):
    """
    Computes Expected Utility and Variance using both models (Eq 7.35 & 7.36).
    """
    # 1. Get Execution Probability
    P_E, _ = predict_execution_prob(acd_params, last_fluct, last_psi, current_price, limit_price)
    
    # 2. Base Utility on Execution
    U_E = current_price - limit_price
    
    # 3. Calculate Expected Utility on Non-Execution via discrete integration
    # Summing over possible future prices j*tick_size (Eq 7.35)
    E_U_NE = 0
    E_U_NE_sq = 0
    
    # Integrate around +/- 3 standard deviations of the GARCH forecast
    sigma = np.sqrt(fitted_garch.forecast(horizon=1).variance.iloc[-1].values[0])
    eval_range = np.arange(limit_price, current_price + (3 * sigma), tick_size)
    
    for p_j in eval_range:
        f_p_j = get_truncated_density(fitted_garch, current_price, limit_price, p_j)
        u_ne_j = current_price - p_j
        
        prob_mass = f_p_j * tick_size
        E_U_NE += u_ne_j * prob_mass
        E_U_NE_sq += (u_ne_j ** 2) * prob_mass
        
    # 4. Total Expected Utility E[U(p^L)]
    E_U = (P_E * U_E) + ((1 - P_E) * E_U_NE)
    
    # 5. Variance V[U(p^L)] (Eq 7.36)
    V_U = (1 - P_E) * (P_E * (U_E - E_U_NE)**2 + E_U_NE_sq - (E_U_NE**2))
    
    # 6. Combined Objective
    objective = E_U - (risk_aversion * V_U)
    
    return P_E, U_E, E_U_NE, E_U, objective

# =====================================================================
# 5. MAIN EXECUTION PIPELINE
# =====================================================================
def run_pipeline(csv_path, tick_size=0.01):
    # Resolve local path relative to this script
    csv_path = os.path.expanduser(csv_path)
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load and prep data
    df = pd.read_csv(csv_path)
    df = prepare_data(df)
    print(f"Loaded data from: {csv_path}")
    
    # Train/Test Split (75/25 as defined in the paper)
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print("Fitting ACD Model on training data...")
    acd_params = fit_acd_model(train_df['fluctuation'].values)
    
    print("Fitting ARMA-GARCH Model on training data...")
    garch_model = fit_arma_garch(train_df['return'].values)
    
    # Setup for Test evaluation
    current_price = test_df['close'].iloc[0]
    last_fluct = train_df['fluctuation'].iloc[-1]
    
    # Reconstruct last PSI for ACD
    psi = np.mean(train_df['fluctuation'])
    for f in train_df['fluctuation'].values:
        psi = acd_params[0] + acd_params[1] * f + acd_params[2] * psi
    
    results = []
    risk_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    print("\nEvaluating T-ACD-ARMA on Test Horizon...")
    for risk in risk_levels:
        limit_price = current_price - tick_size # Example limit price 1 tick down
        
        P_E, E_U_E, E_U_NE, E_U, obj = evaluate_t_acd_arma(
            acd_params, garch_model, last_fluct, psi, current_price, limit_price, risk, tick_size
        )
        
        results.append({
            "Lambda": risk,
            "P_E": round(P_E, 2),
            "E(U_E)": round(E_U_E, 4),
            "E(U_{NE})": round(E_U_NE, 4),
            "E(U)": round(E_U, 4),
            "Objective": round(obj, 4)
        })
        
    # Format exactly like Table 7.3
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("T-ACD-ARMA RESULTS (Mirroring Table 7.3 format)")
    print("="*80)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run T-ACD-ARMA Table 7.3-style analysis on one file or all CSVs in a folder.")
    parser.add_argument("--csv", default=None,
                        help="Path to a single input CSV file (relative or absolute).")
    parser.add_argument("--csv-dir", default="torun",
                        help="Path to a directory containing CSV files to run. Defaults to the local torun directory.")
    parser.add_argument("--tick-size", type=float, default=0.01,
                        help="Tick size for limit pricing and discrete integration.")
    args = parser.parse_args()

    if args.csv:
        run_pipeline(args.csv, tick_size=args.tick_size)
    else:
        csv_dir = os.path.expanduser(args.csv_dir)
        if not os.path.isabs(csv_dir):
            csv_dir = os.path.join(os.path.dirname(__file__), csv_dir)

        if not os.path.isdir(csv_dir):
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

        csv_files = sorted([f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')])
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {csv_dir}")

        for filename in csv_files:
            print(f"\nRunning Table 7.3 analysis for: {filename}")
            run_pipeline(os.path.join(csv_dir, filename), tick_size=args.tick_size)