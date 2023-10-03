import pymc3 as pm              # PyMC3 for Bayesian modeling
import pandas as pd             # Pandas for data manipulation
import numpy as np              # NumPy for numerical operations
import matplotlib.pyplot as plt # Matplotlib for data visualization

# Load your yeast dataset from a CSV file
data = pd.read_csv('Book1.csv')  # Make sure 'data.csv' is in the same directory or provide the full path

# Extract the relevant columns from your dataset
t = data['time'].values         # Extract the 'time' column and convert it to a NumPy array
v = data['volume'].values       # Extract the 'volume' column and convert it to a NumPy array

# Define the PyMC3 model
with pm.Model() as yeast_model:
    # Prior distributions for K and b
    K = pm.Uniform('K', lower=1, upper=20)  # Create a uniform prior for parameter 'K' with bounds
    b = pm.Uniform('b', lower=0, upper=1)   # Create a uniform prior for parameter 'b' with bounds
    
    # Likelihood function (observed data)
    V_pred = K - 0.45 * pm.math.exp(-b * t)
    # Calculate the predicted 'volume' based on the model equation
    
    likelihood = pm.Normal('likelihood', mu=V_pred, sd=0.1, observed=v)
    # Create a likelihood distribution using observed 'volume' data and predicted values
    
    # Perform MCMC sampling
    trace = pm.sample(1000, tune=1000, cores=1)
    # Run MCMC sampling with 1000 iterations and 1000 tuning steps (single-core)

# Posterior analysis and visualization
summary = pm.summary(trace).round(2)
# Generate summary statistics for MCMC samples and round to 2 decimal places

# Plot posterior distributions for K and b
pm.traceplot(trace)
# Create trace plots to visualize the posterior distributions of 'K' and 'b'
plt.show()

# Calculate log-likelihood values for each data point
log_likelihood = -pm.Normal.dist(mu=V_pred, sd=0.1).logp(v).sum()
# Calculate the log-likelihood for the observed 'volume' data based on the model predictions

# Confidence intervals (95%)
ci_95_K = np.percentile(trace['K'], [2.5, 97.5])
# Calculate the 95% confidence interval for parameter 'K'
ci_95_b = np.percentile(trace['b'], [2.5, 97.5])
# Calculate the 95% confidence interval for parameter 'b'

# Report the results
print("Parameter Estimates:")
print(summary)
# Print the summary statistics for parameter estimates
print("\nConfidence Intervals (95%):")
print("K:", ci_95_K)
print("b:", ci_95_b)
# Print the 95% confidence intervals for 'K' and 'b'
print("\nLog-Likelihood Value:")
print(log_likelihood)
# Print the log-likelihood value for the model