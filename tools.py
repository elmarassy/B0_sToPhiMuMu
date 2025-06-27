import jax
import iminuit
from iminuit import Minuit
from jax import jit, vmap
import jax.numpy as jnp

@jit
def gaussDistribution(data, mu, sigma):
    return (1 / jnp.sqrt(2*jnp.pi*sigma**2)) * jnp.exp(-(data - mu)**2/(2*sigma**2))

@jit
def gaussLikelihood(data, mu, sigma):
    vals = vmap(lambda data: gaussDistribution(data, mu, sigma))(data)
    return -jnp.sum(jnp.log(vals))

gradHelper = jax.grad(gaussLikelihood, argnums=[1, 2])

class GaussNLL():

    def __init__(self, data):
        self.data = data
        self.gradientHelper = gradHelper

    def __call__(self, mu, sigma):
        likelihood = gaussLikelihood(self.data, mu, sigma)
        if jnp.isfinite(likelihood):
            return float(likelihood)
        return 1e12

    @property
    def grad(self):
        return lambda *params: self.gradientHelper(self.data, *params)


def fitGaussian(data):
    mu = jnp.mean(data)
    sigma = jnp.std(data)
    nll = GaussNLL(data)
    m = Minuit(nll, mu, sigma)

    m.limits = [[float(jnp.min(data)), float(jnp.max(data))], [0, 2*sigma]]
    m.strategy = 0
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    m.hesse()
    return m.values, m.errors
