import jax.numpy as jnp
from jax import jit, grad, vmap
from jax import random
from jax import lax
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

from slicereparam.rootfinder import dual_bisect_method, choose_start

def swap_axes(xs0, us, norm_ds, xLs, xRs, alphas):
    xs0 = jnp.swapaxes(xs0,0,1)
    us = jnp.swapaxes(us,0,1)
    norm_ds = jnp.swapaxes(norm_ds,0,1)
    xLs = jnp.swapaxes(xLs,0,1)
    xRs = jnp.swapaxes(xRs,0,1)
    alphas = jnp.swapaxes(alphas,0,1)
    return xs0, us, norm_ds, xLs, xRs, alphas

class slicesampler(object):
    """
    A slice sampler object with an unnormalized log pdf and parameters. 
    The object draws reparameterized samples from the unnormalized log pdf.
    TODO: The object computes gradients with respect to parameters. Probably needs a loss? 

    Inputs
    - params : params 
    - log pdf : function that computes log pdf...
    - D : dimensionality of sampling
    - Sc : number of samples per chain
    - num_chains : number of MCMC chains

    """
    def __init__(self, params, log_pdf, D, total_loss=None, Sc=1, num_chains=1, **kwargs):
        
        self.params = params 
        self.log_pdf = log_pdf
        vmapped_log_pdf = jit(vmap(self.log_pdf, (0,None)))

        self.Sc = Sc 
        self.num_chains = num_chains
        self.D = D

        # set up for backwards pass
        # compute necessary gradients
        def log_pdf_theta(theta, x):    return self.log_pdf(x, theta)
        def log_pdf_x(x, theta):        return self.log_pdf(x, theta)
        def log_pdf_ad(x, theta, a, d): return self.log_pdf(x + a * d, theta)
        self.grad_x = jit(grad(log_pdf_x))
        self.grad_theta = jit(grad(log_pdf_theta))
        self.grad_x_ad = jit(grad(log_pdf_ad))

        # grad log normalizer of posterior
        self.vmapped_grad_theta = jit(vmap(self.grad_theta, (None,0)))

        if total_loss is not None:
            self.total_loss = total_loss
            self.loss_grad_xs = jit(grad(total_loss))
            self.loss_grad_params = jit(grad(lambda params, x : total_loss(x, params)))

    def forwards_step(self, x, theta, u1, u2, d):#, aL, bR):
        func = lambda alpha : self.log_pdf(x + alpha * d, theta) - self.log_pdf(x, theta) - jnp.log(u1) # root
        aL, bR = choose_start(func)
        z_L, z_R = dual_bisect_method(func, aL=aL, bL=-1e-10, aR=1e-10, bR=bR)
        x_L = x + d*z_L
        x_R = x + d*z_R
        x = (1 - u2) * x_L + u2 * x_R
        alphas = jnp.array([z_L, z_R])
        return x, x_L, x_R, alphas

    # def vmapped_forwards_step(self, x, theta, u1, u2, d):
        # return vmap(self.forwards_step, (None,0,None,0,0,0))

    def forwards(self, S, theta, x, us, ds):
        xs = jnp.zeros((S+1, self.num_chains, self.D))
        xs = index_update(xs, index[0, :, :], x)
        xLs = jnp.zeros((S, self.num_chains, self.D))
        xRs = jnp.zeros((S, self.num_chains, self.D))
        alphas = jnp.zeros((S, self.num_chains, 2))
        init_val = [xs, xLs, xRs, alphas, x]

        def body_fun(i, val):
            xs, xLs, xRs, alphas, x = val 
            # x, x_L, x_R, alpha = self.vmapped_forwards_step(x, theta, us[i,:,0], us[i,:,1], ds[i])
            x, x_L, x_R, alpha = vmap(self.forwards_step, (0,None,0,0,0))(x, theta, us[i,:,0], us[i,:,1], ds[i])
            xs = index_update(xs, index[i+1, :, :], x)
            xLs = index_update(xLs, index[i, :, :], x_L)
            xRs = index_update(xRs, index[i, :, :], x_R)
            alphas = index_update(alphas, index[i, :, :], alpha)
            val = [xs, xLs, xRs, alphas, x]
            return val

        xs, xLs, xRs, alphas, x = lax.fori_loop(0, S, body_fun, init_val)
        return xs, xLs, xRs, alphas

    # set up randomness
    def generate_randomness(self, key):
        key, *subkeys = random.split(key, 4)
        us = random.uniform(subkeys[0], (self.Sc,self.num_chains,2))
        ds = random.normal(subkeys[1], (self.Sc*self.num_chains,self.D))
        ds_norm = ds / jnp.sqrt(jnp.sum(ds**2, axis=1))[:,None]
        ds_norm = ds_norm.reshape((self.Sc, self.num_chains, self.D))
        x0 = random.normal(subkeys[2], (self.num_chains, self.D))
        return us, ds_norm, x0, key

    @partial(jit, static_argnums=(0))
    def forwards_sample(self, theta, key):
        us, norm_ds, x0, key = self.generate_randomness(key)
        # theta = self.params
        # key, subkey = random.split(key)
        # x0 = theta[:D] + jnp.sqrt(jnp.exp(theta[D:])) * random.normal(subkey, (num_chains, D))
        # x0 = random.normal(subkey, (self.num_chains, self.D))
        xs0, xLs, xRs, alphas = self.forwards(self.Sc, theta, x0, us, norm_ds)
        return xs0, us, norm_ds, xLs, xRs, alphas, key

    def sample_initialization(self):
        """
        Define a function for sampling the initial condition. Would be
        nice to allow for the user to change this.
        """
        return

    def neal_sample(self):
        """
        Also Neal sampling. 
        """
        return

    def backwards_step(self, theta, dL_dtheta, us, d, x, xL, xR, alphas, dL_dx, prev_dL_dx):

        u1 = us[0]
        u2 = us[1]
        z_L = alphas[0]
        z_R = alphas[1]

        # compute loss for current sample
        # set prev_dL_dx to zero at first
        dL_dx_s = dL_dx + prev_dL_dx

        # compute gradients of xL and xR wrt theta
        L_grad_theta = -1.0 * (self.grad_theta(theta, xL) - self.grad_theta(theta, x)) / jnp.dot(d, self.grad_x_ad(x, theta, z_L, d))
        R_grad_theta = -1.0 * (self.grad_theta(theta, xR) - self.grad_theta(theta, x)) / jnp.dot(d, self.grad_x_ad(x, theta, z_R, d))

        # compute gradient dL / dtheta
        dLd = jnp.dot(dL_dx_s, d) # dot product between loss gradient and direction - this is used multiple times 
        dL_dtheta_s = u2 * dLd * R_grad_theta + (1-u2) * dLd * L_grad_theta
        dL_dtheta = dL_dtheta + dL_dtheta_s

        # propagate loss backwards : compute gradient times Jacobian of dx_s  / dx_{s-1}
        L_grad_x = -1.0 * ( self.grad_x_ad(x, theta, z_L, d) - self.grad_x(x, theta) ) / jnp.dot(d, self.grad_x_ad(x, theta, z_L, d))
        R_grad_x = -1.0 * ( self.grad_x_ad(x, theta, z_R, d) - self.grad_x(x, theta) ) / jnp.dot(d, self.grad_x_ad(x, theta, z_R, d))
        prev_dL_dx = dL_dx_s + u2 * dLd * R_grad_x + (1-u2) * dLd * L_grad_x

        return dL_dtheta, prev_dL_dx

    def backwards(self, S, theta, us, ds, xs, xLs, xRs, alphas, dL_dxs):

        dL_dtheta = jnp.zeros_like(theta)
        prev_dL_dx = jnp.zeros_like(xs[0])
        init_val = [S-1, dL_dtheta, prev_dL_dx]

        def cond_fun(val):
            return val[0] > -1

        def body_fun(val):
            s = val[0]
            dL_dtheta, prev_dL_dx = val[1:] 
            dL_dtheta, prev_dL_dx = self.backwards_step(theta, dL_dtheta, us[s,:], ds[s], xs[s], 
                                                xLs[s], xRs[s], alphas[s], dL_dxs[s], prev_dL_dx)
            val[0] -= 1
            return [val[0], dL_dtheta, prev_dL_dx]

        val = lax.while_loop(cond_fun, body_fun, init_val)
        dL_dtheta = val[1]
        return dL_dtheta

    @partial(jit, static_argnums=(0))
    def compute_gradient(self, params, dL_dxs, forwards_out):
        """
        This function computes the gradient given the outputs of forward sampling
        and losses associated with the xs. 

        dL_dxs - gradient of loss for each x, size num_chains x S x D
        """

        # unpack forwards out and "swap axes", ignoring the key
        xs0, us, norm_ds, xLs, xRs, alphas = swap_axes(*forwards_out[:-1])

        # vmapped backwards function
        vmapped_backwards = vmap(self.backwards, (None, None, 0, 0, 0, 0, 0, 0, 0))

        # gradient of params through samples 
        dL_dtheta = jnp.mean(
            vmapped_backwards(self.Sc, params, us, norm_ds, xs0, xLs, xRs, alphas, dL_dxs), 
            axis=0)

        return dL_dtheta

    @partial(jit, static_argnums=(0))
    def estimate_gradient(self, theta, key):
        # self.params = theta
        xs0, us, norm_ds, xLs, xRs, alphas, key = self.forwards_sample(theta, key)
        xs = xs0[-1:].reshape((self.num_chains, self.D), order='F')

        # backwards pass
        dL_dxs = self.loss_grad_xs(xs, theta)
        dL_dxs = dL_dxs.reshape((self.num_chains, 1, self.D))
        dL_dxs = jnp.hstack((jnp.zeros((self.num_chains, self.Sc-1, self.D)), dL_dxs))

        xs0, us, norm_ds, xLs, xRs, alphas = swap_axes(xs0, us, norm_ds, xLs, xRs, alphas)
        vmapped_backwards = jit(vmap(self.backwards, (None, None, 0, 0, 0, 0, 0, 0, 0)))
        dL_dthetas = vmapped_backwards(self.Sc, theta, us, norm_ds, xs0, xLs, xRs, alphas, dL_dxs)
        dL_dtheta = jnp.mean(dL_dthetas, axis=0)
        dL_dtheta = dL_dtheta + self.loss_grad_params(theta, xs) / self.num_chains

        loss = self.total_loss(xs, theta) / self.num_chains
        dL_dtheta = dL_dtheta - jnp.mean(self.vmapped_grad_theta(theta, xs), axis=0)

        return dL_dtheta, loss, key

    # def fit(self, key):
