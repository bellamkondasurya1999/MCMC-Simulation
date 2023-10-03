import numpy as np               # Import NumPy for numerical operations
import pymc3 as pm               # Import PyMC3 for Bayesian modeling
import pandas as pd              # Import Pandas for data manipulation
import theano.tensor as tt       # Import Theano for symbolic math
import matplotlib.pyplot as plt  # Import Matplotlib for data visualization

# Load your "wilson.csv" dataset
# Make sure the CSV file is in the same directory as your Python script or provide the full path
wilson_data = pd.read_csv('wilson.csv')

# Extract the relevant columns from the DataFrame
days = wilson_data['days']      # Extract the 'days' column
weight = wilson_data['weight']  # Extract the 'weight' column

# Constants and parameters
N = 90            # Set an upper bound for N (a parameter in the model)
b = 0.01          # Set an upper bound for b (another parameter in the model)
P0 = 5.0          # Set the initial condition P(0)
delta_t = 1.0     # Set the time step
num_timesteps = len(wilson_data)  # Calculate the number of timesteps based on the data
num_iterations = 1000

# Define the time values
t_values = np.arange(0, num_timesteps * delta_t, delta_t)

# Define the PyMC3 model
with pm.Model() as wilson_model:
    # Prior distributions for N and b
    N_param = pm.Uniform('N', lower=60, upper=N)  # Define 'N' with a uniform prior
    b_param = pm.Uniform('b', lower=0, upper=b)   # Define 'b' with a uniform prior
    
    # Initial condition P(0)
    P0_param = pm.Normal('P0', mu=P0, sd=1)  # Define 'P0' with a normal prior
    
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
    trace = pm.sample(num_iterations, tune=num_iterations, cores=1)  # MCMC sampling with specified iterations

# Print summary statistics for parameter estimates and confidence intervals
print(pm.summary(trace).round(2))

# Print a message indicating the start of the credible intervals section
print("\nConfidence Intervals (95%):")

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
