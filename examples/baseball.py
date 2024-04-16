from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, grad, vmap, random
from jax import random
from jax.flatten_util import ravel_pytree
from jax.scipy import stats
from jax.scipy.special import gammaln, logit, expit, logsumexp
import pandas as pd
import matplotlib.pyplot as plt
import time

from slicereparam.functional import setup_slice_sampler

# import data
# print current directory
from pathlib import Path
import pandas as pd
#%%
print(Path.cwd())
# find parent directory of directory that script is running in
parent_dir = Path(__file__).resolve().parent.parent

d = pd.read_csv(parent_dir / "data/efron-morris-75-data.tsv", sep='\t', lineterminator='\n')
#%%
y = jnp.array(d['Hits'].values) # hits (numbed of successes)
trials = jnp.array(d['At-Bats'].values) # at bats (number of trials)
N = y.shape[0] #number of players
data = jnp.hstack((y[:, None], trials[:, None]))

y_test = jnp.array(d['SeasonHits'].values) - y
trials_test = jnp.array(d['RemainingAt-Bats'].values)

# temp params
log_phi = -1.0
log_kappa = 2.0
logits = -1.0 * jnp.ones((N,))
_x = [log_phi, log_kappa, logits]
x, x_unflatten = ravel_pytree(_x)

def _binomial_logpdf(k, n, p):
    logp = 0.0
    logp += k * jnp.log(p)
    logp += (n-k) * jnp.log(1.0 - p)
    logp += gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
    return logp
binomial_logpdf = vmap(_binomial_logpdf, (0, 0, 0))

k_scale = jnp.array([1.5])
k_scale2 = jnp.array([0.5])

def _heldout_logp(x):
    _, _, logits = x_unflatten(x)
    theta = expit(logits)
    return _binomial_logpdf(y_test, trials_test, theta)
heldout_logp = vmap(_heldout_logp, (0))


# def log_pdf(x, k_scale, data):
def log_pdf(x, k_scale):
    logit_phi, log_kappam1, logits = x_unflatten(x)
    phi = expit(logit_phi)
    theta = expit(logits)
    kappa = jnp.exp(log_kappam1) + 1.0
    logp = 0.0
    logp += stats.uniform.logpdf(phi)
    logp += stats.pareto.logpdf(kappa, k_scale[0], loc=1.0)
    logp += jnp.sum(stats.beta.logpdf(theta, phi * kappa, (1.-phi) * kappa))
    logp += jnp.sum(_binomial_logpdf(y, trials, theta))
    return logp
grad_log_pdf = grad(log_pdf, argnums=(1))
vmap_grad_log_pdf = jit(vmap(grad_log_pdf, (0, None)))

key = random.PRNGKey(13131313)
D = x.shape[0]
S = 4000
num_chains = 10
slice_sample = setup_slice_sampler(log_pdf, D, S, num_chains)


key, *subkeys = random.split(key, 4)
log_phi_init = logit(0.2 + 0.1 * random.uniform(subkeys[0], (num_chains, 1)))
log_kappa_init = 2.0 + 3.0 * random.uniform(subkeys[1], (num_chains, 1))
logits_init = logit(0.25 + 0.1 * random.uniform(subkeys[2], (num_chains, D-2)))
x_in = jnp.hstack((log_phi_init, log_kappa_init, logits_init))

key, subkey = random.split(key)
start_time = time.time()
out = slice_sample(k_scale, x_in, subkey)
print("Slice sample execution time: %s seconds" % (time.time() - start_time))
out2 = out[:, 2000:, :]
xs = out2.reshape((2000*num_chains,D), order='F')
#%%
plt.figure()
plt.plot(xs[:, 2], xs[:, 1], '.')
#%%
plt.figure()
plt.subplot(121)
plt.hist(xs[:,0], bins=35, density=True)
plt.title("logit $\phi$")
plt.xlim([-1.6, -0.2])
plt.subplot(122)
log_km1 = xs[:,1]
kappas = jnp.exp(log_km1) + 1.0
log_kappas = jnp.log(kappas)
plt.hist(log_kappas, bins=35, density=True)
plt.title("log $\kappa$")
plt.xlim([1.0, 5.5])
#%%
def evaluate_ll(xs):
    logps = heldout_logp(xs)
    log_ys_test = logsumexp(logps, axis=0) - jnp.log(xs.shape[0])
    return jnp.sum(log_ys_test)

def meank(k_scale, x_in, key):
    start_time = time.time()
    out = slice_sample(k_scale, x_in, key)
    print("Slice sample execution time: %s seconds" % (time.time() - start_time))
    out = out[:, 2000:, :]
    xs = out.reshape((2000*num_chains,D), order='F')
    return evaluate_ll(xs)
grad_meank = jit(grad(meank))

key, subkey = random.split(key)
start_time = time.time()
out = slice_sample(k_scale, x_in, subkey)
print("Slice sample execution time: %s seconds" % (time.time() - start_time))
out2 = out[:, 2000:, :]
xs = out2.reshape((2000*num_chains,D), order='F')
#%%
dk = 1e-3
k_scale1 = k_scale - dk
k_scale2 = k_scale + dk
start_time = time.time()
out1 = slice_sample(k_scale1, x_in, subkey)
print("Slice sample execution time for k_scale1: %s seconds" % (time.time() - start_time))
out1 = out1[:, 2000:, :]
xs1 = out1.reshape((2000*num_chains,D), order='F')
start_time = time.time()
out2 = slice_sample(k_scale2, x_in, subkey)
print("Slice sample execution time for k_scale2: %s seconds" % (time.time() - start_time))
out2 = out2[:, 2000:, :]
xs2 = out2.reshape((2000*num_chains,D), order='F')
lk_mean1 = evaluate_ll(xs1)
lk_mean2 = evaluate_ll(xs2)
grad_k = (lk_mean2 - lk_mean1) / (2.0 * dk)

grad_val_reparam = grad_meank(k_scale, x_in, subkey)
phi_mean = jnp.mean(expit(xs[:, 0]))
N_samp = xs.shape[0]
cost_x = expit(xs[:, 0])
sample_grads = vmap_grad_log_pdf(xs, k_scale)
grad_val_score = jnp.cov(cost_x, sample_grads[:, 0])[0, 1]

print("FD: ", grad_k)
print("Reparam: ", grad_val_reparam)
print("Score: ", grad_val_score)
#%%
def sample_covariance(x1, x2):
    assert x1.shape[0] == x2.shape[0]
    x1_mean = jnp.mean(x1)
    x2_mean = jnp.mean(x2)
    N_len = x1.shape[0]
    return 1.0 / (N_len-1.0) * jnp.sum((x1 - x1_mean) * (x2 - x2_mean))
vmap_sample_covariance = jit(vmap(sample_covariance, (0, 0)))


S2 = 1000
burn_in = 500
num_chains2 = 1
slice_sample2 = setup_slice_sampler(log_pdf, D, S2, num_chains2)
def meank(k_scale, x_in, key):
    start_time = time.time()
    out = slice_sample2(k_scale, x_in, key)
    print("Slice sample2 execution time: %s seconds" % (time.time() - start_time))
    out = out[:, burn_in:, :]
    xs = out.reshape(((S2-burn_in)*num_chains2,D), order='F')
    return jnp.mean(expit(xs[:, 2]))
grad_meank = jit(grad(meank))

key, subkey = random.split(key)
start_time = time.time()
reparam_grad = grad_meank(k_scale, x_in[:num_chains2], subkey)
print("Slice sample2 execution for reparam grad: %s seconds" % (time.time() - start_time))

out = slice_sample2(k_scale, x_in[:num_chains2], subkey)
out = out[:, burn_in:, :]
xs2 = out.reshape(((S2-burn_in)*num_chains2,D), order='F')

sample_grads = vmap_grad_log_pdf(xs2, k_scale)
# score_grad