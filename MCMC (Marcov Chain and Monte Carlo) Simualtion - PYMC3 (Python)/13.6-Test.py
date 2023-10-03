import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your yeast dataset from a CSV file
data = pd.read_csv('Book1.csv')  # Make sure 'data.csv' is in the same directory or provide the full path

# Extract the relevant columns from your dataset
t = data['time'].values
v = data['volume'].values

# Define the PyMC3 model
with pm.Model() as yeast_model:
    # Prior distributions for K and b
    K = pm.Uniform('K', lower=1, upper=20)
    b = pm.Uniform('b', lower=0, upper=1)
    
    # Likelihood function (observed data)
    V_pred = K - 0.45 * pm.math.exp(-b * t)
    likelihood = pm.Normal('likelihood', mu=V_pred, sd=0.1, observed=v)
    
    # Perform MCMC sampling
    trace = pm.sample(1000, tune=1000, cores=1)

# Posterior analysis and visualization
summary = pm.summary(trace).round(2)

# Plot posterior distributions for K and b
pm.traceplot(trace)
plt.show()

# Calculate log-likelihood values for each data point
log_likelihood = -pm.Normal.dist(mu=V_pred, sd=0.1).logp(v).sum()

# Confidence intervals (95%)
ci_95_K = np.percentile(trace['K'], [2.5, 97.5])
ci_95_b = np.percentile(trace['b'], [2.5, 97.5])

# Report the results
print("Parameter Estimates:")
print(summary)
print("\nConfidence Intervals (95%):")
print("K:", ci_95_K)
print("b:", ci_95_b)
print("\nLog-Likelihood Value:")
print(log_likelihood)

# Import necessary libraries