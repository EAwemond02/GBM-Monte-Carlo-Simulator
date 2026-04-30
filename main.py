'''
Monte Carlo Trading Simulator 3.0
CREATED BY: Connor Nguyen
Date: 4/13/26

Simulates trading account equity using percent risk per trade.
Configurable win rate and reward to risk rate, number of simulations, number of trades per simulation, ruin threshold, and seed generation for reproducibility.
Slider to see how graphs change when volatility/mu change the outputting graphs.
Outputs include average ending price, average drawdown, profit probability, sharpe ratio, percentiles (fat tails), expected shortfalls.
'''
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import yfinance as yf
from options import monte_carlo_call, monte_carlo_put, black_scholes_call, black_scholes_put, calculate_delta, calculate_gamma, calculate_theta, calculate_vega
from simulation import run_simulation
from analytics import calculate_drawdown, calculate_sharpe, calculate_var_es, calculate_prob_of_profit

#Data
tickers = ['SPY']
start_date = '2021-01-01'
data = yf.download(tickers, start=start_date)
Returns = np.log(data['Close']/data['Close'].shift(1))
Returns = Returns.dropna()
mu = Returns.mean().item()
sigma = Returns.std().item()
Starting_Price = data['Close'].iloc[-1].item()
risk_free_rate = 0.04
sigma_annual = sigma * np.sqrt(252)

#Config
paths_count = 1000
trading_days = 252
np.random.seed(67)
print_debug = False
print_summary = True
print_options_summary = True

if print_debug:
    print(mu, sigma)

#Simulate real world (for risk metrics)
price_paths = run_simulation(paths_count, trading_days, mu, sigma, Starting_Price)
final_prices = price_paths[:, -1]

#Simulate Risk neutral (for option pricing)
price_paths_rn = run_simulation(paths_count, trading_days, risk_free_rate/252, sigma, Starting_Price)
final_prices_rn = price_paths_rn[:, -1]

#Options
strike = Starting_Price  # At-the-money option
T = 1
h = 1
option_type = 'call'
call_price = monte_carlo_call(final_prices_rn, Starting_Price, risk_free_rate, T)
put_price =  monte_carlo_put(final_prices_rn, Starting_Price, risk_free_rate, T)
bs_call = black_scholes_call(Starting_Price, strike, T, risk_free_rate, sigma_annual)
bs_put =  black_scholes_put(Starting_Price, strike, T, risk_free_rate, sigma_annual)
call_at_S1 =     black_scholes_call(Starting_Price + 1, strike, T, risk_free_rate, sigma_annual)
call_at_P1 =     black_scholes_call(Starting_Price, strike, T, risk_free_rate, sigma_annual + 0.01)
call_at_T1 =     black_scholes_call(Starting_Price, strike, T - (1/252), risk_free_rate, sigma_annual)
C_up   = black_scholes_call(Starting_Price + h, strike, T, risk_free_rate, sigma_annual)
C_mid  = black_scholes_call(Starting_Price,     strike, T, risk_free_rate, sigma_annual)
C_down = black_scholes_call(Starting_Price - h, strike, T, risk_free_rate, sigma_annual)

#Options Calculations
analytical_delta = calculate_delta(Starting_Price, strike, T, risk_free_rate, sigma_annual, option_type)
numerical_delta = call_at_S1 - bs_call
vega = calculate_vega(call_at_P1, bs_call)
gamma = calculate_gamma(C_up, C_mid, C_down, h)
theta = calculate_theta(call_at_T1, bs_call)

#Analytics
sharpe = calculate_sharpe(mu, sigma, risk_free_rate)
var, es = calculate_var_es(final_prices)
pop = calculate_prob_of_profit(final_prices, Starting_Price)
drawdowns = [calculate_drawdown(path) for path in price_paths]
percentiles = np.quantile(final_prices, [0.05, 0.50, 0.95])

#Percentile path bands
p5 = np.percentile(price_paths, 5, axis=0)
p50 = np.percentile(price_paths, 50, axis=0)
p95 = np.percentile(price_paths, 95, axis=0)

if print_summary:
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"VaR (5th): {var:,.2f}")
    print(f"Expected Shortfall: {es:,.2f}")
    print(f"Avg Drawdown: {np.mean(drawdowns):.2%}")
    print(f"Worst Drawdown: {np.max(drawdowns):.2%}")
    print(f"Probability of Profit: {pop:.2%}")

if print_options_summary:
    print(f"BS Call Price:  {bs_call:,.2f}")
    print(f"MC Call Price:  {call_price:,.2f}")
    print(f"BS Put Price:   {bs_put:,.2f}")
    print(f"MC Put Price:   {put_price:,.2f}")
    print(f"Analytical Delta: {analytical_delta:,.2f}")
    print(f"Numerical Delta: {numerical_delta:,.2f}")
    print(f"Vega: ${vega:,.2f}")
    print(f"Gamma: {gamma:.5f}")
    print(f"Theta: {theta:.2f}")



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
plt.subplots_adjust(bottom=0.25)
for i in range(paths_count):
    ax1.plot(price_paths[i])
ax1.set_title('Equity Curve vs Trading Days')
ax1.set_ylabel('Value')
ax1.set_xlabel('Days')

ax2.hist(final_prices, bins=30, edgecolor='black')
ax2.set_title('Distribution of Final SPY Prices')
ax2.set_xlabel('Final Price')
ax2.set_ylabel('Frequency')

ax3.plot(p5, color='red', label='5th Percentile (Bear)')
ax3.plot(p50, color='blue', label='50th Percentile (Base)')
ax3.plot(p95, color='green', label='95th Percentile (Bull)')
ax3.legend()
ax3.set_title('Percentile Bands')
ax3.set_xlabel('Trading Days')
ax3.set_ylabel('Price')
ax3.fill_between(range(trading_days + 1), p5, p95, alpha=0.1, color='blue')

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Volatility', 0.05, 0.50, valinit=sigma)

def update(val):
    new_sigma = slider.val

    #rerun calculations
    Z = np.random.normal(size=(paths_count, trading_days))
    daily_returns = np.exp((mu - 0.5 * new_sigma**2) + new_sigma * Z)
    price_paths = np.zeros((paths_count, trading_days + 1))
    price_paths[:, 0] = Starting_Price
    for t in range(trading_days):
        price_paths[:, t+1] = price_paths[:, t] * daily_returns[:, t]
    
    #Redraw lines
    ax1.clear()
    for i in range(paths_count):
        ax1.plot(price_paths[i])
    ax1.set_title('Equity Curve vs Trading Days')
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Days')

    final_prices = price_paths[:, -1]
    p5 = np.percentile(price_paths, 5, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)

    ax2.clear()
    ax2.hist(final_prices, bins=30, edgecolor='black')
    ax2.set_title('Distribution of Final SPY Prices')
    ax2.set_xlabel('Final Price')
    ax2.set_ylabel('Frequency')

    ax3.clear()
    ax3.plot(p5, color='red', label='5th Percentile (Bear)')
    ax3.plot(p50, color='blue', label='50th Percentile (Base)')
    ax3.plot(p95, color='green', label='95th Percentile (Bull)')
    ax3.fill_between(range(trading_days + 1), p5, p95, alpha=0.1, color='blue')
    ax3.legend()
    ax3.set_title('Percentile Bands')
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Price')
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()