import numpy as np
import pymc3 as pm
import pandas as pd
import theano.tensor as tt
import matplotlib.pyplot as plt

# Load your "wilson.csv" dataset
# Make sure the CSV file is in the same directory as your Python script or provide the full path
wilson_data = pd.read_csv('wilson.csv')

# Extract the relevant columns from the DataFrame
days = wilson_data['days']
weight = wilson_data['weight']

# Constants and parameters
N = 90  # Upper bound for N
b = 0.01  # Upper bound for b
P0 = 5.0  # Initial condition P(0)
delta_t = 1.0  # Time step
num_timesteps = len(wilson_data)  # Number of timesteps based on the data
num_iterations = 1000

# Define the time values
t_values = np.arange(0, num_timesteps * delta_t, delta_t)

# Define the PyMC3 model
with pm.Model() as wilson_model:
    # Prior distributions for N and b
    N_param = pm.Uniform('N', lower=60, upper=N)
    b_param = pm.Uniform('b', lower=0, upper=b)
    
    # Initial condition P(0)
    P0_param = pm.Normal('P0', mu=P0, sd=1)
    
    # Differential equation model
    def mass_growth(N=N_param, b=b_param, P0=P0_param):
        P = [P0]
        for i in range(1, num_timesteps):
            dP_dt = b * (N - P[-1])
            P.append(P[-1] + dP_dt * delta_t)
        return tt.stack(P)
    
    # Calculate the mass of the dog over time
    P = pm.Deterministic('P', mass_growth())
    
    # Likelihood function (observed data)
    likelihood = pm.Normal('likelihood', mu=P, sd=0.1, observed=weight)
    
    # Perform MCMC sampling
    trace = pm.sample(num_iterations, tune=num_iterations, cores=1)

# Print summary statistics for parameter estimates and confidence intervals
print(pm.summary(trace).round(2))
# 
# 
# 
# 

# Print a message indicating the start of the credible intervals section
print("\nConfidence Intervals (95%):")

# 
# 
# 
# 

# Plot posterior distributions for N and b
pm.traceplot(trace)
plt.show()

# Plot the posterior distribution of P0
pm.plot_posterior(trace, var_names=['P0'])
plt.show()

# Calculate log-likelihood values
with wilson_model:
    ppc = pm.sample_posterior_predictive(trace, samples=1000)
log_likelihood_values = np.log(ppc['likelihood'])

# Print the mean log-likelihood
print(f"Log-Likelihood Mean: {log_likelihood_values.mean():.2f}")
