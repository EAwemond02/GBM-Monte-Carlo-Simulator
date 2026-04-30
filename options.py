import numpy as np
from scipy.stats import norm

def monte_carlo_call(final_prices, strike, risk_free_rate, T=1):
    payoffs = np.maximum(final_prices - strike, 0)
    payoff_mean = payoffs.mean()
    discounted = payoff_mean * np.exp(-risk_free_rate * T)
    return discounted

def monte_carlo_put(final_prices, strike, risk_free_rate, T=1):
    payoffs = np.maximum(strike - final_prices, 0)
    payoff_mean = payoffs.mean()
    discounted = payoff_mean * np.exp(-risk_free_rate * T)
    return discounted

def black_scholes_call(S, K, T, r, sigma):
    d1 = ((np.log(S/K) + (r + 0.5 * (sigma**2)) * T) / (sigma * np.sqrt(T)))
    d2 = d1 - (sigma * np.sqrt(T))
    return (S * norm.cdf(d1) ) - (K * np.exp(-r * T) * norm.cdf(d2))

def black_scholes_put(S, K, T, r, sigma):
    d1 = ((np.log(S/K) + (r + 0.5 * (sigma**2)) * T) / (sigma * np.sqrt(T)))
    d2 = (d1 - (sigma * np.sqrt(T)))
    return (K * np.exp(-r * T) * norm.cdf(-d2) - (S * norm.cdf(-d1)))

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1
    
def calculate_gamma(C_up, C_mid, C_down, h): return (C_up - 2 * C_mid + C_down) / h**2
def calculate_theta(call_at_T1, bs_call): return (call_at_T1 - bs_call)
def calculate_vega(call_at_P1, bs_call): return (call_at_P1 - bs_call)