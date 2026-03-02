import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
# Load the Excel file shown in your directory image
file_name = "data/AKBA_1min_RTH_40days.csv"
df = pd.read_csv(file_name)

# Parse the datetime column and set it as the index
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Calculate 1-minute returns for the ARMA-GARCH model
df['return'] = np.log(df['close'] / df['close'].shift(1))

# Calculate Maximum Price Fluctuation (M_T) for Limit Buy Orders (Open - Low)
df['fluctuation_buy'] = df['open'] - df['low']
df.dropna(inplace=True)

# Intraday Seasonality component s(t_i) 
df['time'] = df.index.time
seasonality = df.groupby('time')['fluctuation_buy'].mean()
df['seasonality'] = df['time'].map(seasonality)

# Deseasonalize the fluctuations for the ACD model
df['deseasonalized_fluctuation'] = df['fluctuation_buy'] / df['seasonality']

# Chronological Train/Test Split (First 30 days train, Last 10 days test)
train_size = int(len(df) * 0.75)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# ==========================================
# 2. MODEL TRAINING (IN-SAMPLE)
# ==========================================
print("Training ARMA-GARCH and ACD Models...")

# Train ARMA(1,0)-GARCH(1,1) on returns
# We scale returns by 100 for numerical stability during MLE optimization
train_returns = train_df['return'] * 100
garch_model = arch_model(train_returns, mean='AR', lags=1, vol='Garch', p=1, q=1)
garch_res = garch_model.fit(disp='off')

# Train ACD Model for Execution Probability
# Note: Standard libraries lack the complex ABAMACD with Generalized Gamma. 
# We use an AR(1) model on deseasonalized fluctuations as a standard baseline proxy.
acd_model = ARIMA(train_df['deseasonalized_fluctuation'], order=(1,0,0))
acd_res = acd_model.fit()

## ==========================================
# 3. OUT-OF-SAMPLE TESTING & OPTIMIZATION
# ==========================================
print("Running Mean-Variance Optimization Backtest...")

risk_aversion_lambda = 0.5 
# FIX 1: Adjust tick size for $1.65 stocks based on Image 1
tick_size = 0.001 
distances = np.arange(1, 21) * tick_size # Test placing orders 0.001 to 0.020 away

profits = []
penalties = [25]

for i in range(1, len(test_df)):
    current_bar = test_df.iloc[i]
    prev_bar = test_df.iloc[i-1]
    
    # FIX 2: Anchor the immediate market price to the current Open
    p_0_M = current_bar['open'] 
    prev_close = prev_bar['close']
    
    # --- A. Forecast Non-Execution Risk (ARMA-GARCH) ---
    forecasts = garch_res.forecast(horizon=1, align='origin')
    mu_f = forecasts.mean.iloc[-1].values / 100.0
    var_f = forecasts.variance.iloc[-1].values / 10000.0
    sigma_f = np.sqrt(var_f)
    
    # --- B. Forecast Execution Probability (ACD Proxy) ---
    expected_deseasonalized_fluc = acd_res.predict(start=len(train_df)+i, end=len(train_df)+i).values
    expected_fluc = expected_deseasonalized_fluc * current_bar['seasonality']
    
    lambda_param = 1 / expected_fluc if expected_fluc > 0 else 1e-6
    
    # FIX 3: Correct Execution Probability to the Survival Function
    P_E_array = np.exp(-lambda_param * distances)
    
    # --- C. Mean-Variance Utility Maximization ---
    best_utility = -np.inf
    best_distance = distances
    
    for idx, delta in enumerate(distances):
        p_L = p_0_M - delta 
        P_E = P_E_array[idx]
        
        U_E = delta 
        
        # Calculate required return relative to the previous close for ARMA-GARCH
        limit_return_req = (p_L - prev_close) / prev_close
        z = (limit_return_req - mu_f) / sigma_f
        
        denominator = 1 - norm.cdf(z)
        inverse_mills = norm.pdf(z) / denominator if denominator > 0 else 0
        
        expected_penalty_return = mu_f + sigma_f * inverse_mills
        expected_p_T = prev_close * (1 + expected_penalty_return)
        
        # U_NE compares market ordering at the open vs market ordering at the close
        U_NE = p_0_M - expected_p_T
        
        E_U = P_E * U_E + (1 - P_E) * U_NE
        V_U = P_E * (1 - P_E) * (U_E - U_NE)**2
        
        Objective = E_U - risk_aversion_lambda * V_U
        
        if Objective > best_utility:
            best_utility = Objective
            best_distance = delta
            
    # --- D. Evaluate the Optimal Strategy Against Reality ---
    optimal_p_L = p_0_M - best_distance
    actual_low = current_bar['low']
    actual_close = current_bar['close']
    
    if actual_low <= optimal_p_L:
        profits.append(p_0_M - optimal_p_L)
    else:
        penalties.append(p_0_M - actual_close)

# ==========================================
# 4. RESULTS
# ==========================================
total_profit = sum(profits)
total_penalty = sum(penalties)

print("\n--- Backtest Results ---")
print(f"Total Trading Intervals: {len(test_df)-1}")
print(f"Successful Fills: {len(profits)}")
print(f"Failed Fills (Market Crosses): {len(penalties)}")
print(f"Total Profit from Fills: ${total_profit:.4f}")
print(f"Total Penalty from Misses: ${total_penalty:.4f}")
print(f"Net Strategy Profit: ${total_profit + total_penalty:.4f}")