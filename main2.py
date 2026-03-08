import pandas as pd
import numpy as np
from arch import arch_model
import warnings

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def calculate_optimal_limit_price(current_price, mu_forecast, var_forecast, signal):
    sigma = np.sqrt(var_forecast)
    if signal == 'BUY':
        limit_price = current_price * np.exp(mu_forecast - 1.0 * sigma)
    else: 
        limit_price = current_price * np.exp(mu_forecast + 1.0 * sigma)
    return limit_price

def update_arma_garch(returns_series):
    rescaled_returns = returns_series * 1000
    am = arch_model(rescaled_returns, mean='AR', lags=1, vol='Garch', p=1, q=1, dist='normal')
    res = am.fit(update_freq=0, disp='off', show_warning=False)
    forecasts = res.forecast(horizon=1)
    
    mu_next = forecasts.mean.iloc[-1].values[0] / 1000
    var_next = forecasts.variance.iloc[-1].values[0] / 1000000
    return mu_next, var_next

# ==========================================
# 2. MAIN TRADING SIMULATION WITH PnL
# ==========================================

def run_trading_simulation(df, signals):
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna().copy()
    
    unique_days = df['Date'].unique()
    training_days = unique_days[:10]   
    trading_days = unique_days[10:]    
    
    history_df = df[df['Date'].isin(training_days)].copy()
    trade_results = []
    
    for current_day in trading_days:
        day_data = df[df['Date'] == current_day].copy()
        
        signal = signals.get(current_day, None)
        if not signal:
            history_df = pd.concat([history_df, day_data])
            continue
            
        # 90 Minutes = 18 bars (since each bar is 5 consecutive minutes)
        morning_data = day_data.iloc[:18]
        trading_window = day_data.iloc[18:]
        
        current_history = pd.concat([history_df, morning_data])
        
        try:
            mu_forecast, var_forecast = update_arma_garch(current_history['Return'])
        except Exception:
            mu_forecast = current_history['Return'].mean()
            var_forecast = current_history['Return'].var()
        
        price_at_min_90 = morning_data.iloc[-1]['Close']
        limit_price = calculate_optimal_limit_price(price_at_min_90, mu_forecast, var_forecast, signal)
        
        executed = False
        exec_price = None
        exec_time = None
        
        for index, row in trading_window.iterrows():
            if signal == 'BUY':
                if row['Low'] <= limit_price:
                    exec_price = limit_price
                    exec_time = row['datetime']
                    executed = True
                    break 
            elif signal == 'SELL':
                if row['High'] >= limit_price:
                    exec_price = limit_price
                    exec_time = row['datetime']
                    executed = True
                    break 
        
        # PnL Calculation based on the end-of-day close
        day_close = day_data['Close'].iloc[-1]
        pnl = 0.0
        
        if executed:
            if signal == 'BUY':
                pnl = day_close - exec_price  # Profit if close is higher than buy price
            else:
                pnl = exec_price - day_close  # Profit if close is lower than sell price
                
        trade_results.append({
            'Date': current_day,
            'Signal': signal,
            'Limit_Price': limit_price,
            'Executed': executed,
            'Exec_Price': exec_price,
            'Day_Close': day_close,
            'PnL': pnl
        })
        
        history_df = pd.concat([history_df, day_data])

    return pd.DataFrame(trade_results)

# ==========================================
# 3. CSV LOAD & CONSECUTIVE CHUNKING
# ==========================================
if __name__ == "__main__":
    
    # 1. Load the dataset
    df_raw = pd.read_csv('data/ARGX_1min_RTH_40days.csv')
    
    # 2. Standardize columns and dates
    df_raw.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_raw = df_raw.sort_values('datetime').reset_index(drop=True)
    df_raw['Date'] = df_raw['datetime'].dt.date
    
    # 3. Group strictly by consecutive 5 rows (bypassing time-gaps)
    df_raw['group_id'] = df_raw.index // 5
    df = df_raw.groupby('group_id').agg({
        'datetime': 'first', # Timestamp of the start of the 5-min block
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Date': 'first'
    }).reset_index(drop=True)
    
    # 4. Identify trading days and mock the signals
    unique_days = df['Date'].unique()
    trading_days = unique_days[10:]
    
    np.random.seed(42) 
    mock_signals = {day: np.random.choice(['BUY', 'SELL']) for day in trading_days}
    
    print("Running backtest with consecutive 5-min chunking...")
    results = run_trading_simulation(df, mock_signals)
    
    # ==========================================
    # 4. PnL SUMMARY
    # ==========================================
    total_trades = len(results)
    executed_trades = results['Executed'].sum()
    total_pnl = results['PnL'].sum()
    win_count = len(results[results['PnL'] > 0])
    loss_count = len(results[results['PnL'] < 0])
    
    print("\n--- DAILY TRADING LOG ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results.to_string(index=False))
    
    print("\n" + "="*40)
    print("          PROPER PnL SUMMARY")
    print("="*40)
    print(f"Total Trading Days Checked : {total_trades}")
    print(f"Orders Successfully Filled : {executed_trades}")
    print(f"Orders Missed (No Fill)    : {total_trades - executed_trades}")
    print("-" * 40)
    print(f"Winning Trades             : {win_count}")
    print(f"Losing Trades              : {loss_count}")
    print(f"Total Cumulative PnL       : {total_pnl:.4f} per share")
    if executed_trades > 0:
        print(f"Average PnL per Exec Trade : {(total_pnl / executed_trades):.4f} per share")
    print("="*40)