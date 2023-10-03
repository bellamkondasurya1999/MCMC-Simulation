import pymc3 as pm              # PyMC3 for Bayesian modeling
import pandas as pd             # Pandas for data manipulation
import numpy as np              # NumPy for numerical operations
import theano.tensor as tt      # Theano for symbolic math
import matplotlib.pyplot as plt # Matplotlib for data visualization

# Load your "phosphorus.csv" dataset
# Make sure the CSV file is in the same directory as your Python script or provide the full path
phosphorus_data = pd.read_csv('phosphorous.csv')

# Extract relevant columns from the DataFrame
daphnia = phosphorus_data['daphnia']  # Extract the 'daphnia' column
algae = phosphorus_data['algae']      # Extract the 'algae' column

# Step 1: Define the PyMC3 model and parameters
with pm.Model() as phos_model:
    # Parameters with bounds
    c = pm.Uniform('c', lower=0, upper=2)     # Create a uniform distribution parameter 'c'
    theta = pm.Uniform('theta', lower=1, upper=20)  # Create a uniform distribution parameter 'theta'

    # Model definition using a custom Theano function
    def custom_power(x, y):
        return tt.pow(x, y)  # Define a custom function for exponentiation using Theano

    daphnia_mean = c * custom_power(algae, 1 / theta)  # Define the mean using custom function and parameters

    # Likelihood function (observed data)
    likelihood = pm.Poisson('daphnia_obs', mu=daphnia_mean, observed=daphnia)
    # Define the likelihood that links the observed data 'daphnia' to the model

# Step 2: Define MCMC settings
phos_iter = 10000  # Number of iterations for MCMC sampling

# Step 3: Compute MCMC estimate
with phos_model:
    trace = pm.sample(phos_iter, cores=1)  # Perform MCMC sampling with specified iterations (single-core)

# Step 4: Analyze results and visualize
summary = pm.summary(trace).round(2)  # Generate summary statistics for parameter estimates

# Display the summary statistics
print("\nSummary Statistics for Parameter Estimates:")
print(summary)

# Create trace plots to visualize the posterior distributions of 'c' and 'theta'
pm.traceplot(trace)
plt.show()

# Generate posterior distribution plots for 'c' and 'theta'
pm.plot_posterior(trace, var_names=['c', 'theta'])
plt.show()

# Optional: Save the trace for further analysis
pm.save_trace(trace, 'phos_trace.pkl', overwrite=True)
print("\nMCMC Trace saved as 'phos_trace.pkl'.")

# Display the generated plots
print("\nPlots displayed.")