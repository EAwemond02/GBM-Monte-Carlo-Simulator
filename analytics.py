import numpy as np

def calculate_drawdown(equity_curve):
    equity_curve = np.array(equity_curve)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (running_peak - equity_curve) / running_peak
    return np.max(drawdowns)

def calculate_sharpe(mu, sigma, risk_free_rate=0.04):
    return ((mu * 252) - risk_free_rate) / (sigma * np.sqrt(252))

def calculate_var_es(final_prices, confidence=0.05):
    var = np.quantile(final_prices, confidence)
    es = final_prices[final_prices < var].mean()
    return var, es

def calculate_prob_of_profit(final_prices, Starting_Price):
    return len(final_prices[final_prices > Starting_Price]) / len(final_prices)