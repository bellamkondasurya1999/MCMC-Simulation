# Import necessary libraries
import pymc3 as pm              # PyMC3 for Bayesian modeling
import pandas as pd             # Pandas for data manipulation
import numpy as np              # NumPy for numerical operations
import matplotlib.pyplot as plt # Matplotlib for data visualization

# Load your dataset from a CSV file
data = pd.read_csv('Book1.csv')  # Make sure 'data.csv' is in the same directory or provide the full path

# Extract the relevant columns from your dataset
t = data['time'].values         # Extract the 'time' column and convert it to a NumPy array
v = data['volume'].values       # Extract the 'volume' column and convert it to a NumPy array

# Define the PyMC3 model
with pm.Model() as yeast_model:
    # Prior distributions for K and b
    K = pm.Uniform('K', lower=1, upper=20)  # Create a uniform prior for parameter 'K' with bounds
    b = pm.Uniform('b', lower=0, upper=1)   # Create a uniform prior for parameter 'b' with bounds
    
    # Calculate 'a' based on the provided relationship
    a = pm.Deterministic('a', pm.math.log(K / 0.45 - 1))  # Calculate 'a' deterministically based on 'K'
    
    # Likelihood function (observed data)
    V_pred = K / (1 + pm.math.exp(a - b * t))  # Calculate the predicted 'volume' based on the model
    likelihood = pm.Normal('likelihood', mu=V_pred, sd=0.1, observed=v)
    # Create a likelihood distribution using observed 'volume' data and predicted values
    
    # Perform MCMC sampling
    trace = pm.sample(1000, tune=1000, cores=1)  # Perform MCMC sampling with specified iterations and tuning steps

# Posterior analysis and visualization
summary = pm.summary(trace).round(2)  # Generate summary statistics for MCMC samples

# Plot posterior distributions for K and b
pm.traceplot(trace)  # Create trace plots to visualize the posterior distributions of 'K' and 'b'
plt.show()  # Display the trace plots

# Plot the posterior distribution of 'a'
pm.plot_posterior(trace, var_names=['a'])  # Generate posterior distribution plots for 'a'
plt.show()  # Display the posterior distribution plot for 'a'

# Calculate log-likelihood values for each data point
log_likelihood = -pm.Normal.dist(mu=V_pred, sd=0.1).logp(v).sum()
# Calculate the log-likelihood for the observed 'volume' data based on the model predictions

# Confidence intervals (95%) for parameters
ci_95_K = np.percentile(trace['K'], [2.5, 97.5])  # Calculate the 95% CI for parameter 'K'
ci_95_b = np.percentile(trace['b'], [2.5, 97.5])  # Calculate the 95% CI for parameter 'b'
ci_95_a = np.percentile(trace['a'], [2.5, 97.5])  # Calculate the 95% CI for parameter 'a'

# Report the results
print("Parameter Estimates:")
print(summary)  # Print the summary statistics for parameter estimates
print("\nConfidence Intervals (95%):")
print("K:", ci_95_K)  # Print the 95% CI for parameter 'K'
print("b:", ci_95_b)  # Print the 95% CI for parameter 'b'
print("a:", ci_95_a)  # Print the 95% CI for parameter 'a'
print("\nLog-Likelihood Value:")
print(log_likelihood)  # Print the log-likelihood value for the model