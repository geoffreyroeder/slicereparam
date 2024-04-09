import os
import pickle
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp 
from jax import jit, grad, vmap
from jax import random
from jax.flatten_util import ravel_pytree
from jax.lax import stop_gradient

import numpy as np

import matplotlib.pyplot as plt
from tqdm.auto import trange
from tqdm import tqdm
import seaborn as sns 
from slicereparam.functional import setup_slice_sampler_with_args


RESULTS_PATH = "/home/groeder/projects/reparam_MALA/code/github/slicereparam/results/"
PLOT_PATH = "/home/groeder/projects/reparam_MALA/code/github/slicereparam/figs/"

# create paths if they don't exist
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)

# set up randomness
seed = 1234
key = random.PRNGKey(seed)

# Set up params
D = 20      # number of latent / data dimensions
N = 1000    # number of data points

# set up slice sampler
S = 128
num_chains = S + 0
S_init = 30
S_grad = 20 

# learning rate params
a0 = 0.05
gam = 0.0001

num_iters = 1000

exp_name = f"toy_encoder_decoder_D{D}_N{N}_S{S}_numc{num_chains}_Sinit{S_init}_Sgrad{S_grad}_seed{seed}_a0{a0}_gam{gam}_niter{num_iters}"
data_file_path = RESULTS_PATH+exp_name+".pkl"

# generate data

@jit
def gaussian_log_pdf(x, mu, Sigma):
    out = -0.5 * (x - mu).T @ jnp.linalg.inv(Sigma) @ (x - mu)
    out = out - 0.5 *  jnp.log(jnp.linalg.det(Sigma))
    out = out - D / 2.0 * jnp.log(2.0 * jnp.pi)
    return out
vmap_gaussian_log_pdf = vmap(gaussian_log_pdf, (0, 0, None))

@jit
def _log_pdf(z, params, x):
    A, b, mu = params
    z_mean = A@x + b
    # q(z|x) = N(Ax + b, 2/3 I)
    return gaussian_log_pdf(z, z_mean, 2.0 / 3.0 * jnp.eye(D))

key, *subkeys = random.split(key, 4)
mu_true = random.normal(subkeys[0], (D,))
z_true = mu_true + random.normal(subkeys[1], (N, D))
x_true = z_true + random.normal(subkeys[2], (N, D))

# init params
key, *subkeys = random.split(key, 4)
mu = random.normal(subkeys[0], (D,))
A  = random.normal(subkeys[1], (D, D))
b  = random.normal(subkeys[2], (D, ))

_params = [A, b, mu]
params, unflatten = ravel_pytree(_params)
log_pdf = jit(lambda z, params, x: _log_pdf(z, unflatten(params), x))
vmapped_log_pdf = jit(vmap(log_pdf, (0, None, 0)))

# slice reparam loss and gradient
@jit
def negative_elbo(zs, xs, params):
    A, b, mu = unflatten(params)
    out = jnp.mean(-0.5 * (zs - mu[None,:])**2) # p(z) = N(mu, I)
    params = stop_gradient(params)
    out = out + jnp.mean(vmap_gaussian_log_pdf(xs, zs, jnp.eye(D))) # p(x|z) = N(z, I)
    out = out - jnp.mean(vmapped_log_pdf(zs, params, xs)) # entropy term (grad part)
    return -1.0 * out

# one for initialization, one for computing gradients
slice_sample_init = setup_slice_sampler_with_args(log_pdf, D, S_init, num_chains)
slice_sample = setup_slice_sampler_with_args(log_pdf, D, S_grad, num_chains)

@jit
def loss_slice(params, z0, xs, key):
    zs_all = slice_sample(params, z0, xs, key)
    zs = zs_all[:, -1, :]
    # loss = negative_elbo(zs, xs, params)
    _, _, mu = unflatten(params)
    out = jnp.mean(jnp.sum(-0.5 * (zs - mu[None,:])**2, axis=1)) # p(z) = N(mu, I)
    params = stop_gradient(params)
    out = out + jnp.mean(vmap_gaussian_log_pdf(xs, zs, jnp.eye(D))) # p(x|z) = N(z, I)
    out = out - jnp.mean(vmapped_log_pdf(zs, params, xs)) # entropy term (grad part)
    return -1.0 * out 

grad_slice = jit(grad(loss_slice))


@jit
def generate_data_idx(key):
    key, subkey = random.split(key)
    data_idx = random.randint(subkey, (S, ), 0, N)
    return data_idx, key



# optimize parameters!
theta = params+0.0
M = theta.shape[0]
losses = [0.0]
thetas = [theta]

try:
    print("Attempting to load results")
    print(f"File path: {data_file_path}")
    with open(data_file_path, "rb") as f:
        data = pickle.load(f)
        thetas_plot = data["thetas"]
    print("Results loaded")
except:
    print("No results found")
    print("Running experiment")

    # pbar = trange(num_iters)
    # pbar.set_description("Loss: {:.1f}".format(losses[0]))

    for i in tqdm(range(num_iters)):

        data_idx, key = generate_data_idx(key)
        ys = x_true[data_idx]

        # forwards_out = model.forwards_sample(theta, key, ys=ys)

        # # process output
        # key = forwards_out[-1]
        # xs0 = forwards_out[0] 
        # xs = xs0[:,-1:,:].reshape((num_chains, D), order='F') # samples for loss
        # dL_dxs = loss_grad_xs(xs, ys, theta)

        # # compute gradient
        # dL_dtheta = model.compute_gradient_one_sample(theta, dL_dxs, forwards_out)
        # dL_dtheta = dL_dtheta + loss_grad_params(theta, xs, ys) / S
        key, *subkeys = random.split(key, 3)
        x0 = random.normal(subkeys[0], (num_chains, D))
        xs = slice_sample_init(theta, x0, ys, subkeys[1])
        x0 = xs[:, -1, :]
        key, subkey = random.split(key, 2)
        dL_dtheta = grad_slice(theta, x0, ys, subkey)

        # TODO - combine grad and val in one function
        losses.append(negative_elbo(xs[:, -1, :], ys, theta))

        # update params
        alpha_t = a0 / (1 + gam * (i+1)) # learning rate 
        theta = theta - dL_dtheta * alpha_t
        thetas.append(theta)
        # if i == 0:
        #     print(f"Loss: {losses[-1]}")
        # pbar.set_description("Loss: {:.1f}".format(losses[-1]))
        # pbar.update()

    # pbar.close()

    # save to file
    with open(data_file_path, "wb") as f:
        data = {
            "thetas": thetas,
        }
        pickle.dump(data, f)


    thetas_plot = np.array(thetas)
    # thetas_reparam_plot = jnp.array(thetas_reparam)
    sns.set_context("talk")
    A_fit, b_fit, mu_fit = unflatten(theta)
    mustar = jnp.mean(x_true, axis=0)
    Astar = jnp.eye(D) / 2 
    bstar = mustar / 2.0
    plt.figure(figsize=[12,4])
    plt.subplot(131)
    plt.imshow(jnp.vstack((mustar, mu_fit, mu)).T, aspect="auto")
    plt.xticks([0.0, 1.0, 2.0], ["$\mu^*$", "$\hat{\mu}$", "$\mu_{init}$"])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(jnp.vstack((bstar, b_fit, b)).T, aspect="auto")
    plt.xticks([0.0, 1.0, 2.0], ["$b^*$", "$\hat{b}$", "$b_{init}$"])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(jnp.vstack((Astar, A_fit, A)).T, aspect="auto")
    plt.xticks([10.0, 30.0, 50.0], ["$A^*$", "$\hat{A}$", "$A_{init}$"])
    plt.yticks([])
    plt.colorbar()
    plt.tight_layout()

    # save plot
    plt.savefig(PLOT_PATH+f"A_b_mu_plot_{exp_name}.pdf")
