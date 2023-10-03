import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt

# Load your "phosphorus.csv" dataset
# Make sure the CSV file is in the same directory as your Python script or provide the full path
phosphorus_data = pd.read_csv('phosphorous.csv')

# Extract relevant columns from the DataFrame
daphnia = phosphorus_data['daphnia']
algae = phosphorus_data['algae']

# Step 1: Define the PyMC3 model and parameters
with pm.Model() as phos_model:
    # Parameters with bounds
    c = pm.Uniform('c', lower=0, upper=2)
    theta = pm.Uniform('theta', lower=1, upper=20)

    # Model definition using a custom Theano function
    def custom_power(x, y):
        return tt.pow(x, y)

    daphnia_mean = c * custom_power(algae, 1 / theta)

    # Likelihood function (observed data)
    likelihood = pm.Poisson('daphnia_obs', mu=daphnia_mean, observed=daphnia)

# Step 2: Define MCMC settings
phos_iter = 10000  # Number of iterations

# Step 3: Compute MCMC estimate
with phos_model:
    trace = pm.sample(phos_iter, cores=1)  # Use cores=1 for single-core execution

# Step 4: Analyze results and visualize
pm.summary(trace).round(2)  # Print summary statistics for parameter estimates

# Plot posterior distributions for parameters c and theta
pm.traceplot(trace)

# Plot the posterior distribution of parameters c and theta
pm.plot_posterior(trace, var_names=['c', 'theta'])

# Optional: Save the trace for further analysis
pm.save_trace(trace, 'phos_trace.pkl',overwrite=True)

# Show the plots
plt.show()