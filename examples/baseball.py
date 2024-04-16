from functools import partial
from pathlib import Path
import pandas as pd
import os
import jax.numpy as jnp
from jax import grad, jit, random, vmap
from jax.flatten_util import ravel_pytree
from jax.scipy.special import expit, gammaln, logit, logsumexp
from jax.scipy import stats
from jax import config
import time
import matplotlib.pyplot as plt
from slicereparam.functional import setup_slice_sampler

config.update("jax_enable_x64", True)
parent_dir = Path(__file__).resolve().parent.parent
RESULTS_DIR = parent_dir / "results"  # Define the directory to save results
data_path = parent_dir / "data/efron-morris-75-data.tsv"

def save_results(data, file_name):
    path = os.path.join(RESULTS_DIR, file_name)
    with open(path, 'wb') as f:
        jnp.save(f, data)

def load_results_or_compute(file_name, compute_func, *args, **kwargs):
    path = os.path.join(RESULTS_DIR, file_name)
    if os.path.exists(path):
        print(f"Loading saved results from {file_name}")
        with open(path, 'rb') as f:
            return jnp.load(f)
    else:
        print(f"No saved results, computing for {file_name}")
        result = compute_func(*args, **kwargs)
        print(f"Saving results to {file_name}")
        save_results(result, file_name)
        return result

data = pd.read_csv(data_path, sep='\t', lineterminator='\n')

# Extract data from the dataset
hits = jnp.array(data['Hits'].values)
at_bats = jnp.array(data['At-Bats'].values)
num_players = hits.shape[0]
season_hits_test = jnp.array(data['SeasonHits'].values) - hits
remaining_at_bats_test = jnp.array(data['RemainingAt-Bats'].values)

# Initialize parameters
log_phi_init = -1.0
log_kappa_init = 2.0
logits_init = -1.0 * jnp.ones((num_players,))
_params = [log_phi_init, log_kappa_init, logits_init]
params, params_unflatten = ravel_pytree(_params)

# Define the binomial log-likelihood function
def _binomial_logpdf(k, n, p):
    logp = 0.0
    logp += k * jnp.log(p)
    logp += (n-k) * jnp.log(1.0 - p)
    logp += gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
    return logp

binomial_logpdf = vmap(_binomial_logpdf, (0, 0, 0))

# Define the Pareto distribution shape parameter
pareto_shape = jnp.array([1.5])

# Define the held-out log-likelihood function
def _heldout_logp(params):
    _, _, logits = params_unflatten(params)
    theta = expit(logits)
    return _binomial_logpdf(season_hits_test, remaining_at_bats_test, theta)

heldout_logp = vmap(_heldout_logp, (0))

# Define the log-posterior function
def log_posterior(params, pareto_shape):
    logit_phi, log_kappam1, logits = params_unflatten(params)
    phi = expit(logit_phi)
    theta = expit(logits)
    kappa = jnp.exp(log_kappam1) + 1.0
    logp = 0.0
    logp += stats.uniform.logpdf(phi)
    logp += stats.pareto.logpdf(kappa, pareto_shape[0], loc=1.0)
    logp += jnp.sum(stats.beta.logpdf(theta, phi * kappa, (1.-phi) * kappa))
    logp += jnp.sum(_binomial_logpdf(hits, at_bats, theta))
    return logp

grad_log_posterior = grad(log_posterior, argnums=(1))
vmap_grad_log_posterior = jit(vmap(grad_log_posterior, (0, None)))

# Set up slice sampler
key = random.PRNGKey(13131313)
num_params = params.shape[0]
num_samples = 20000
num_chains = 10
slice_sample = setup_slice_sampler(log_posterior, num_params, num_samples, num_chains)

key, *subkeys = random.split(key, 4)
log_phi_init = logit(0.2 + 0.1 * random.uniform(subkeys[0], (num_chains, 1)))
log_kappa_init = 2.0 + 3.0 * random.uniform(subkeys[1], (num_chains, 1))
logits_init = logit(0.25 + 0.1 * random.uniform(subkeys[2], (num_chains, num_params-2)))
params_init = jnp.hstack((log_phi_init, log_kappa_init, logits_init))

def run_slice_sample_and_save(key):
    key, subkey = random.split(key)
    start_time = time.time()
    slice_sample_output = slice_sample(pareto_shape, params_init, subkey)
    print("Slice sample execution time: %s seconds" % (time.time() - start_time))
    return slice_sample_output

slice_sample_output_file_name = "slice_sample_output_{}_{}_{}.npy".format(pareto_shape[0], num_chains, num_samples)
key, subkey = random.split(key)
slice_sample_output = load_results_or_compute(slice_sample_output_file_name, run_slice_sample_and_save, subkey)

# Evaluate held-out log-likelihood
def evaluate_heldout_ll(params_samples):
    logps = heldout_logp(params_samples)
    log_ys_test = logsumexp(logps, axis=0) - jnp.log(params_samples.shape[0])
    return jnp.sum(log_ys_test)

# Estimate sensitivity of the first moment of phi to the Pareto shape parameter
def mean_phi(pareto_shape, params_init, key):
    slice_sample_output = slice_sample(pareto_shape, params_init, key)
    slice_sample_output = slice_sample_output[:, burn_in:, :]
    params_samples = slice_sample_output.reshape((remaining*num_chains, num_params), order='F')
    return evaluate_heldout_ll(params_samples)

grad_mean_phi = jit(grad(mean_phi))

key, subkey = random.split(key)
final_slice_sample_output_file_name = "final_slice_sample_output_{}_{}_{}.npy".format(pareto_shape[0], num_chains, num_samples)
slice_sample_output = load_results_or_compute(final_slice_sample_output_file_name, run_slice_sample_and_save, subkey)

# Estimate sensitivity of the second moment of phi to the Pareto shape parameter
num_samples2 = 1000
burn_in = 500
num_chains2 = 1
slice_sample2 = setup_slice_sampler(log_posterior, num_params, num_samples2, num_chains2)

def mean_phi_squared(pareto_shape, params_init, key):
    slice_sample_output = slice_sample2(pareto_shape, params_init, key)
    slice_sample_output = slice_sample_output[:, burn_in:, :]
    params_samples = slice_sample_output.reshape(((num_samples2-burn_in)*num_chains2, num_params), order='F')
    return jnp.mean(expit(params_samples[:, 2]))

grad_mean_phi_squared = jit(grad(mean_phi_squared))

# Plotting functions
def plot_logit_phi_and_log_kappa(params_samples):
    plt.figure()
    plt.subplot(121)
    plt.hist(params_samples[:,0], bins=35, density=True)
    plt.title("logit $\phi$")
    plt.xlim([-1.6, -0.2])
    plt.subplot(122)
    log_km1 = params_samples[:,1]
    kappas = jnp.exp(log_km1) + 1.0
    log_kappas = jnp.log(kappas)
    plt.hist(log_kappas, bins=35, density=True)
    plt.title("log $\kappa$")
    plt.xlim([1.0, 5.5])
    plt.tight_layout()
    plt.savefig("logit_phi_and_log_kappa.png")
    plt.close()

# Printing functions
def print_gradient_estimates(reparam_grad, score_grad, fd_grad):
    print("Reparameterization gradient: ", reparam_grad)
    print("Score function gradient: ", score_grad)
    print("Finite difference gradient: ", fd_grad)

# Plot logit phi and log kappa
burn_in = int(0.1*num_samples)
slice_sample_output = slice_sample_output[:, burn_in:, :]
remaining = num_samples - burn_in
params_samples = slice_sample_output.reshape((remaining*num_chains, num_params), order='F')
plot_logit_phi_and_log_kappa(params_samples)

# Estimate gradients
key = random.PRNGKey(13131313)
key, subkey = random.split(key)
dk = 1e-3
pareto_shape1 = pareto_shape - dk
pareto_shape2 = pareto_shape + dk 

params_samples1_file_name = "params_samples1_{}_{}_{}.npy".format(pareto_shape1[0], num_chains, num_samples)
key, subkey = random.split(key)
params_samples1 = load_results_or_compute(params_samples1_file_name, slice_sample, pareto_shape1, params_init, subkey)
params_samples1 = params_samples1[:, burn_in:, :].reshape((remaining*num_chains, num_params), order='F')

params_samples2_file_name = "params_samples2_{}_{}_{}.npy".format(pareto_shape2[0], num_chains, num_samples)
key, subkey = random.split(key)
params_samples2 = load_results_or_compute(params_samples2_file_name, slice_sample, pareto_shape2, params_init, subkey)
params_samples2 = params_samples2[:, burn_in:, :].reshape((remaining*num_chains, num_params), order='F')

ll_mean1 = evaluate_heldout_ll(params_samples1)
ll_mean2 = evaluate_heldout_ll(params_samples2)
fd_grad = (ll_mean2 - ll_mean1) / (2.0 * dk)

key, subkey = random.split(key)
reparam_grad = grad_mean_phi(pareto_shape, params_init, subkey)

sample_grads = vmap_grad_log_posterior(params_samples, pareto_shape)
score_grad = jnp.cov(expit(params_samples[:, 0]), sample_grads[:, 0])[0, 1]

# Print gradient estimates
print_gradient_estimates(reparam_grad, score_grad, fd_grad)


# Define the range of alpha values
alpha_vals = jnp.arange(1.0, 2.05, 0.05)

# Compute the local sensitivity estimates for each alpha value using vmap
local_sensitivities_fn = vmap(lambda alpha: grad_mean_phi(jnp.array([alpha]), params_init, subkey))
local_sensitivities = local_sensitivities_fn(alpha_vals)

# Compute the hierarchical mean hit probability for each alpha value using vmap
mean_hit_probs_fn = vmap(lambda alpha: mean_phi(jnp.array([alpha]), params_init, subkey))
mean_hit_probs = mean_hit_probs_fn(alpha_vals)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(alpha_vals, mean_hit_probs, 'o-', label='Hierarchical mean hit probability')
plt.plot(alpha_vals, local_sensitivities, 'o-', label='Local sensitivity estimate')
plt.xlabel(r'Pareto shape parameter $\alpha$')
plt.ylabel('Value')
plt.title('Local sensitivity of hierarchical mean hit probability')
plt.legend()
plt.tight_layout()
plt.savefig("local_sensitivity_plot.png")
plt.close()