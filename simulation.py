import numpy as np

def run_simulation(paths_count, trading_days, mu, sigma, Starting_Price):
    Z = np.random.normal(size=(paths_count, trading_days))
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
    price_paths = np.zeros((paths_count, trading_days + 1))
    price_paths[:, 0] = Starting_Price
    for t in range(trading_days):
        price_paths[:, t+1] = price_paths[:, t] * daily_returns[:, t]
    return price_paths