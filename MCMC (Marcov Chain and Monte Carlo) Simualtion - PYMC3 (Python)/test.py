import numpy as np
import pymc3 as pm
import pandas as pd
import theano.tensor as tt
import matplotlib.pyplot as plt

# Create a DataFrame from your data
data = pd.DataFrame({'days': [1, 2, 3, 4, 5, 6],
                     'weight': [6.25, 10, 20, 23, 26, 27.6]})

# Extract the relevant columns from the DataFrame
days = data['days']
weight = data['weight']

# Constants and parameters
N = 90  # Upper bound for N
b = 0.01  # Upper bound for b
P0 = 5.0  # Initial condition P(0)
delta_t = 1.0  # Time step
num_timesteps = len(data)
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

# Posterior analysis and visualization
pm.summary(trace).round(2)

# Plot posterior distributions for N and b
pm.traceplot(trace)
plt.show()

# Plot the posterior distribution of P0
pm.plot_posterior(trace, var_names=['P0'])
plt.show()
# 
# 
# 