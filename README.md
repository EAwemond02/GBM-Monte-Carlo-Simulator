Overview:
This project simulates future price paths of SPY (Or other stocks) using Geometric Brownian Motion (GBM) and analyzez risk through Monte Carlo simulation.
It provides metrics such as VaR, Expected Shortfall, Sharpe Ratio, Drawdowns, and Probability of Profit.

Methodology:
The program uses log returns because of the log addition property. The sum of two logarithms is equal to the product of their arguments.
This then allows us to simply add returns over time, instead of multiplying everything.
The primary formula for calculating prices is the Geometric Brownian formula. 
S1 = S0 * e^((mu - 0.5(Variance)) - Volatility * Z)
Mu or drift is calculated by taking the mean return of the time frame we are analyzing. 
Sigma or Volatility is calculated by taking the standard deviation of the time frame we are analyzing.
For the time frame, this program uses Jan 1, 2021 as the starting date, and today's date as the ending date.
Z is generated for every day using NumPy's random.normal function, which generates values from a normal/gaussian distribution.
Monte Carlo is a distribution of outcomes and not a single forecast because we are running thousands of paths instead of one singular forecast.

Feature/Metrics:
It computes and stores all values that are calculated throughout the simulation.
It calculates and stores percentile bands, currently using the 5th, 50th, and 95th percentile.
Histogram of final prices to see distribution.
VaR & Expected Shortfall. VaR is the final price of the 95th percentile run. While Expected shortfall is the mean of all final values that are equal or less than the expected shortfall.
Sharpe Ratio. Follows the regular Sharpe Ratio formula, with the risk-free rate being 0.04, matching that of most treasury bonds.
Max Drawdown. Calculates the lowest low after a previous high. The program displays both the average worst drawdown, as well as the worst drawdown throughout all the runs.
Probability of Profit. Finds the percentage of runs that finish above the starting point.
Interactive Volatility Slider. Reruns and changes the graphs based on the volatility of the user's choice.
<img width="1489" height="698" alt="image" src="https://github.com/user-attachments/assets/56d695cb-11c9-4ad7-9218-561ba317cc8b" />

Key Insights:
As observed in the graph, the distribution is right-skewed. The GBM assumes returns are normally distributed, which implies the prices will follow a log-normal distribution.
Since prices cannot fall below zero but can grow without bound, the upside tail is longer than the downside, showing positive skew.
Higher volatility widens percentile bands significantly. This is because higher volatility can result in more extreme gains and losses, leading to more paths walking a lot higher or lower, widening the bands.
Expected Shortfall highlights tail risk beyond VaR. The VaR is more of something that can determine if this was just an average day or a bad day. 
However expected shortfall shows you what the average losses on these bad days are.
Sharpe ratio decreases when volatility increases, holding drift constant. The sharpe ratio uses returns as a metric, and the returns change highly based on volatility.

Limitations:
Assume constant volatility (no GARCH)
Assumes independent returns (no autocorrelation)
Uses historical drift (not risk-neutral)
No jumps (no jump diffusion)
Cannot show regime changes

Slider in action:
0.05 Volatility: <img width="1395" height="593" alt="image" src="https://github.com/user-attachments/assets/ee54958c-59a6-48e1-973e-8e4128812669" />
0.1 Volatility: <img width="1437" height="596" alt="image" src="https://github.com/user-attachments/assets/4fe54c9a-afaa-44a0-ae51-32efb75cdd01" />
0.2 Volatility: <img width="1433" height="581" alt="image" src="https://github.com/user-attachments/assets/06cecc66-1a6f-44e1-9234-60d6dd424f54" />




Option Pricing Methodology

The engine calculates European Option prices through two different mathematical lenses to ensure model robustness.

Analytical Method: Black-Scholes-Merton
Using the standard BSM closed-form solution:
C = S_0 N(d_1) - K e^{-rT} N(d_2)$$
d_1 = np.exp((mu - 0.5 * sigma**2) + sigma * Z)

Numerical Method: Monte Carlo Simulation
The system simulates thousands of geometric Brownian motion paths under a risk-neutral measure (where drift = risk-free rate) to calculate the expected discounted payoff. This method provides the flexibility to handle more complex terminal payoffs beyond the standard European structure.

The Greeks
- Delta: Analytical Delta and Numerical delta.
- Gamma: Central difference method with an underlying price bump.
- Vega: Sensitivity to a 1% shift in annualized volatility.
- Theta: Daily time decay calculated via a 1/252 time step.
