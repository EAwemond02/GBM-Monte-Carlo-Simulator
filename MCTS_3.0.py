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
import pandas as pd

paths_count = 100
trading_days = 252
tickers = ['SPY']
start_date = '2021-01-01'
np.random.seed(67)
data = yf.download(tickers, start=start_date)
print_debug = False
print_summary = True

#Log Returns
Returns = np.log(data['Close']/data['Close'].shift(1))
Returns = Returns.dropna()

mu = Returns.mean().item()
sigma = Returns.std().item()
Starting_Price = data['Close'].iloc[-1].item()
if print_debug:
    print(mu, sigma)

#Simulate
Z = np.random.normal(size=(paths_count, trading_days))
daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
price_paths = np.zeros((paths_count, trading_days + 1))
price_paths[:, 0] = Starting_Price
for t in range(trading_days):
    price_paths[:, t+1] = price_paths[:, t] * daily_returns[:, t]

#Final price and Percentile
final_prices = price_paths[:, -1]
percentiles = [.05, .5, .95]
percentile_calculations = []
for i in range(len(percentiles)):
    percentile_calculations = np.quantile(final_prices, percentiles)
var = np.quantile(final_prices, 0.05)
expected_shortfall = final_prices[final_prices < var].mean()

#Sharpe Ratio
sharpe = ((mu * 252) - 0.04) / (sigma * np.sqrt(252))

#Drawdown
def calculate_drawdown(equity_curve):
    equity_curve = np.array(equity_curve)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (running_peak - equity_curve) / running_peak
    return np.max(drawdowns)
drawdowns = [calculate_drawdown(path) for path in price_paths]

#Percentile path bands
p5 = np.percentile(price_paths, 5, axis=0)
p50 = np.percentile(price_paths, 50, axis=0)
p95 = np.percentile(price_paths, 95, axis=0)

if print_summary:
    print(
        f"5th Percentile: {percentile_calculations[0]:,.2f}\n"
        f"50th Percentile: {percentile_calculations[1]:,.2f}\n"
        f"95th Percentile: {percentile_calculations[2]:,.2f}\n"
        f"Expected Shortfall: {expected_shortfall:,.2f}\n"
        f"Sharpe Ratio: {sharpe:,.3f}\n"
        f"Average drawdown: {np.mean(drawdowns):,.2%}\n"
        f"Worst Drawdown: {np.max(drawdowns):,.2%}\n"
        f"Probability of Profit: {len(final_prices[final_prices > Starting_Price])/paths_count:,.2%}\n"
    )


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