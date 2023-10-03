import pymc3 as pm              # PyMC3 for Bayesian modeling
import pandas as pd             # Pandas for data manipulation
import numpy as np              # NumPy for numerical operations
import theano.tensor as tt      # Theano for symbolic math
import matplotlib.pyplot as plt # Matplotlib for data visualization
import time                     # Time for measuring execution time

# Load your "phosphorus.csv" dataset
phosphorus_data = pd.read_csv('phosphorous.csv')

# Extract relevant columns from the DataFrame
daphnia = phosphorus_data['daphnia']  # Extract the 'daphnia' column
algae = phosphorus_data['algae']      # Extract the 'algae' column

# Create an empty list to store execution times
execution_times = []

# Define a list of iteration counts to test
iteration_counts = [1, 10, 100, 1000, 10000]

# Loop through each iteration count
for phos_iter in iteration_counts:
    # Define a PyMC3 model for Bayesian analysis
    with pm.Model() as phos_model:
        # Parameters with bounds
        c = pm.Uniform('c', lower=0, upper=2)             # Create a uniform distribution parameter 'c'
        theta = pm.Uniform('theta', lower=1, upper=20)    # Create a uniform distribution parameter 'theta'

        # Model definition using a custom Theano function
        def custom_power(x, y):
            return tt.pow(x, y)  # Define a custom function for exponentiation using Theano

        daphnia_mean = c * custom_power(algae, 1 / theta)  # Define the mean using custom function and parameters

        # Likelihood function (observed data)
        likelihood = pm.Poisson('daphnia_obs', mu=daphnia_mean, observed=daphnia)
        # Define the likelihood that links the observed data 'daphnia' to the model

    # Measure the start time for MCMC sampling
    start_time = time.time()

    # Run MCMC sampling
    with phos_model:
        trace = pm.sample(phos_iter, cores=1)  # Perform MCMC sampling with specified iterations (single-core)

    # Measure the end time for MCMC sampling
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Append the execution time to the list
    execution_times.append(execution_time)

    # Print the execution time for the current iteration count
    print(f"Iterations: {phos_iter}, Time (seconds): {execution_time:.2f}")

# Create a scatterplot to visualize execution times vs. iteration counts
plt.scatter(iteration_counts, execution_times)
plt.xlabel('Number of Iterations')
plt.ylabel('Time (seconds)')
plt.title('MCMC Parameter Estimation Time vs. Iterations')
plt.grid(True)

# Show the plot
plt.show()

# Display execution times for each iteration count
for i in range(len(iteration_counts)):
    print(f"Iterations: {iteration_counts[i]}, Time (seconds): {execution_times[i]:.2f}")
