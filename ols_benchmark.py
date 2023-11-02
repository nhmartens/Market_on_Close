import numpy as np
import random
import time

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def OLS(x, y, intercept=True):
    if intercept:
        X = np.ones((x.shape[0], x.shape[1]+1))
        X[:,1:] = x
    coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T),y)
    return coef


# Generate Random Data
np.random.seed(1)
X = np.array([np.random.randn(100_000), np.random.randn(100_000)]).T
y = 5 + 2 * X[:,0] + 3 * X[:,1] + np.random.randn(100_000)

# Statsmodels implementation
no_runs = 100
start_time = time.perf_counter()
for _ in range(no_runs):
    a = sm.OLS(y, sm.add_constant(X)).fit()
end_time = time.perf_counter()
print(f'Statsmodels: {(end_time - start_time) / no_runs}')


# Sklearn Implementation
regressionModel = LinearRegression()
y = y.reshape(-1,1)
for _ in range(no_runs):
    b = regressionModel.fit(X, y)
end_time = time.perf_counter()
print(f'SK-Learn: {(end_time - start_time) / no_runs}')


# Custom Implementation
for _ in range(no_runs):
    c = OLS(X, y.reshape(-1,1))
end_time = time.perf_counter()
print(f'Custom: {(end_time - start_time) / no_runs}')


# On MacOS with an ARM64 Chip and my working environment, the Statsmodel performs the best by far
# This is not really intuitive as it calculates a lot of stats in the background
