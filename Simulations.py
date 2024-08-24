import numpy as np

def generate_gbm_data(S0, mu, sigma, T, dt):
    """
    Generates data following a Geometric Brownian Motion (GBM) process.

    Parameters:
    - S0: Initial stock price
    - mu: Drift coefficient (expected return)
    - sigma: Volatility (standard deviation of returns)
    - T: Total time (in years)
    - dt: Time step (in years)

    Returns:
    - time_steps: Array of time steps
    - prices: Array of simulated prices
    """

    # Number of steps
    N = int(T / dt)
    
    # Time steps
    time_steps = np.linspace(0, T, N)
    
    # Initialize the price array
    prices = np.zeros(N)
    prices[0] = S0
    
    # Generate the random component of the motion (Wiener process)
    W = np.random.normal(0, np.sqrt(dt), N)
    
    # Simulate the GBM process
    for t in range(1, N):
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * W[t])

    return time_steps, prices

import matplotlib.pyplot as plt

# Parameters
S0 = 100    # Initial stock price
mu = 0.05   # Expected return (5% per year)
sigma = 0.2 # Volatility (20% per year)
T = 2.0     # Total time (1 year)
dt = 0.01   # Time step (1 day)

# Generate data
time_steps, prices = generate_gbm_data(S0, mu, sigma, T, dt)

#Or suppose we have a portfolio, meaning each param is a list.
S0 = [100,34.4,103,57.1,77]
mu = [0.05,0.06,0.03,0.04,0.04]
sigma = [0.2,0.25,0.1,0.15,0.15]
T = 2.0
dt = 0.01
portfolio = []
for i in range(len(S0)):
    time_steps,prices = generate_gbm_data(S0[i],mu[i],sigma[i],T,dt)
    portfolio.append(prices[-1])
print(portfolio)

portfolio_return = 0
procentual_return = 0
for j in range(len(S0)):
    portfolio_return+=portfolio[j]-S0[j]

print(portfolio_return)

#Calculation the historical volatility per asset:
mean = []
volatility = []
std = 0

for i in range(len(S0)):
    time_steps,prices = generate_gbm_data(S0[i],mu[i],sigma[i],T,dt)
    mean.append(np.mean(prices))
    for j in range(len(prices)):
        std+=np.sqrt((1/(len(prices)-1))*(prices[j]-mean)**2)
    volatility.append(std)
    std = 0
volatility=volatility[-1]
print(volatility)