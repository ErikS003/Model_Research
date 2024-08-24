import scipy
import numpy as np
from scipy import stats
def Beta(portfolio,market):
    slope, intercept, r_value, p_value, std_err = stats.linregress(market, portfolio)
    beta_regression = slope
    return beta_regression


def sharpe_ratio(returns, risk_free_rate=0.0):
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_dev = np.std(returns, ddof=1)  # Using ddof=1 for sample standard deviation
    sharpe_ratio = mean_excess_return / std_dev
    
    return sharpe_ratio
