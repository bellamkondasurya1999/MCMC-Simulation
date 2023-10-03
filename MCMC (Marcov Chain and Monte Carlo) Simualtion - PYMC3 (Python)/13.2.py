import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt
import time

# Load your "phosphorus.csv" dataset
# Make sure the CSV file is in the same directory as your Python script or provide the full path
phosphorus_data = pd.read_csv('phosphorous.csv')

# Extract relevant columns from the DataFrame
daphnia = phosphorus_data['daphnia']
algae = phosphorus_data['algae']

# Create an empty list to store execution times
execution_times = []

# Define a list of iteration counts to test
iteration_counts = [1, 10, 100, 1000, 10000]

for phos_iter in iteration_counts:
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

    start_time = time.time()
    with phos_model:
        trace = pm.sample(phos_iter, cores=1)  # Use cores=1 for single-core execution
    end_time = time.time()

    execution_time = end_time - start_time
    execution_times.append(execution_time)

# Create a scatterplot
plt.scatter(iteration_counts, execution_times)
plt.xlabel('Number of Iterations')
plt.ylabel('Time (seconds)')
plt.title('MCMC Parameter Estimation Time vs. Iterations')
plt.grid(True)

# Show the plot
plt.show()

# Display execution times
for i in range(len(iteration_counts)):
    print(f"Iterations: {iteration_counts[i]}, Time (seconds): {execution_times[i]:.2f}")
