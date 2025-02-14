import numpy as np
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import minimize
import numpy as np

class EWMA:
    def __init__(self, lambda_: float = 0.94):
        """
        Initializes the EWMA model.
        
        :param lambda_: Decay factor, typically between 0 and 1. Default is 0.94.
        """
        self.lambda_ = lambda_

    def calculate_volatility(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate the EWMA volatility for a series of returns.
        
        :param returns: Array of returns.
        :return: Array of EWMA volatilities.
        """
        n = len(returns)
        volatilities = np.zeros(n)
        volatilities[0] = np.sqrt(np.mean(returns**2))  # Start with the standard deviation of returns

        for t in range(1, n):
            volatilities[t] = np.sqrt(self.lambda_ * volatilities[t-1]**2 + (1 - self.lambda_) * returns[t-1]**2)
        
        return volatilities

# Example usage:
returns = np.random.randn(100) / 100  # Simulated returns
ewma_model = EWMA(lambda_=0.94)
ewma_volatility = ewma_model.calculate_volatility(returns)
print(ewma_volatility[:5])

class GARCH:
    def __init__(self):
        """
        Initializes the GARCH(1,1) model.
        """
        self.omega = None
        self.alpha = None
        self.beta = None

    def _garch_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculates the negative log-likelihood of the GARCH(1,1) model.
        
        :param params: Array of GARCH parameters [omega, alpha, beta].
        :param returns: Array of returns.
        :return: Negative log-likelihood.
        """
        omega, alpha, beta = params
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)  # Initialize with variance of returns

        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

        log_likelihood = -(-0.5 * np.sum(np.log(sigma2) + (returns**2) / sigma2))
        return log_likelihood

    def fit(self, returns: np.ndarray):
        """
        Fit the GARCH(1,1) model to the returns data.
        
        :param returns: Array of returns.
        """
        # Initial guess for the parameters
        initial_params = np.array([0.000001, 0.1, 0.85])
        
        # Constraints to ensure stationarity and positivity
        bounds = [(0, None), (0, 1), (0, 1)]

        # Minimize the negative log-likelihood
        result = minimize(self._garch_likelihood, initial_params, args=(returns,), bounds=bounds)

        self.omega, self.alpha, self.beta = result.x

    def calculate_volatility(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate the GARCH(1,1) volatility for a series of returns.
        
        :param returns: Array of returns.
        :return: Array of GARCH(1,1) volatilities.
        """
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)  # Initialize with variance of returns

        for t in range(1, n):
            sigma2[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * sigma2[t-1]
        
        return np.sqrt(sigma2)

# Example usage:

#print(returns)
# Fit the GARCH(1,1) model
returns = np.random.randn(100) / 100  # Simulated returns
garch_model = GARCH()
garch_model.fit(returns)
garch_volatility = garch_model.calculate_volatility(returns)
# Forecast volatility

# Plot the forecasted volatility
import matplotlib.pyplot as plt
print(garch_volatility[:5])
print(np.mean(garch_volatility[:5]))
plt.plot(garch_volatility, label="Forecasted Volatility")
plt.title("GARCH(1,1) Forecasted Volatility")
plt.xlabel("Time")
plt.ylabel("Volatility")
plt.legend()
plt.show()

