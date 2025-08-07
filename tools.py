import os
os.environ["JAX_ENABLE_X64"] = "True"
import jax
from iminuit import Minuit
from jax import jit, vmap
import jax.numpy as jnp
from torchquad import Simpson, set_up_backend
from scipy.stats import chi2
set_up_backend("jax", data_type="float64")
integrator = Simpson()


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


def pearsonCoefficient(x, y):
    EX = jnp.mean(x)
    EY = jnp.mean(y)
    cov = jnp.mean((x - EX) * (y - EY))
    stdX = jnp.std(x)
    stdY = jnp.std(y)
    return cov / (stdX * stdY)


def pearsonCorrelation(X):
    X_T = X.T
    correlation = vmap(lambda xi: vmap(lambda xj: pearsonCoefficient(xi, xj))(X_T))(X_T)
    return correlation


@jit
def erf(z):
    sign = jnp.sign(jnp.real(z))
    sign = jnp.where(sign == 0, 1, sign)
    z = sign * z
    def helper(x, c21, c11, b21, b11):
        return (x**2 + c21*x + c11) / (x**2 + b21*x + b11)
    term0 = (0.56418958354775629 / (z + 2.06955023132914151))
    term1 = helper(z, 2.71078540045147805, 5.80755613130301624, 3.47954057099518960, 12.06166887286239555)
    term2 = helper(z, 3.47469513777439592, 12.07402036406381411, 3.72068443960225092, 8.44319781003968454)
    term3 = helper(z, 4.00561509202259545, 9.30596659485887898, 3.90225704029924078, 6.36161630953880464)
    term4 = helper(z, 5.16722705817812584, 9.12661617673673262, 4.03296893109262491, 5.13578530585681539)
    term5 = helper(z, 5.95908795446633271, 9.19435612886969243, 4.11240942957450885, 4.48640329523408675)
    poly = term0 * term1 * term2 * term3 * term4 * term5
    erf_approx_pos_re = 1.0 - poly * jnp.exp(-z * z)
    final_result = sign * erf_approx_pos_re
    return final_result

@jit
def exponentialDistribution(var, decay, expRange):
    return jnp.exp(-decay*var) / (jnp.exp(-decay*expRange[0]) - jnp.exp(-decay*expRange[1])) * decay


@jit
def convolutionIntegral(gamma, z, t):
    sigma = (0.0375 + t/2000.0)
    oscillation = 0.5 * (1 + erf(
        (1/jnp.sqrt(2)) * (t/sigma + gamma*sigma*(z-1))
    )) * jnp.exp(t*gamma*(z-1) + (gamma * sigma * (z-1))**2 / 2)
    return oscillation


def getIntegrals(acceptance, timeRange, gamma, x, y):
    coshIntegral = integrator.integrate(lambda time: acceptance(time) * ((convolutionIntegral(gamma, y, time) + convolutionIntegral(gamma, -y, time))/2.0),
                                        dim=1, N=999999, integration_domain=[timeRange])
    sinhIntegral = integrator.integrate(lambda time: acceptance(time) * ((convolutionIntegral(gamma, y, time) - convolutionIntegral(gamma, -y, time))/2.0),
                                        dim=1, N=999999, integration_domain=[timeRange])
    cosIntegral = integrator.integrate(lambda time: acceptance(time) * jnp.real(convolutionIntegral(gamma, 1j*x, time)),
                                       dim=1, N=999999, integration_domain=[timeRange])
    sinIntegral = integrator.integrate(lambda time: acceptance(time) * jnp.imag(convolutionIntegral(gamma, 1j*x, time)),
                                       dim=1, N=999999, integration_domain=[timeRange])
    return coshIntegral, sinhIntegral, cosIntegral, sinIntegral

def poissonError(n):
    alpha = 1 - 0.6827
    if n == 0:
        lower = 0
        upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (n + 1))
    else:
        lower = 0.5 * chi2.ppf(alpha / 2, 2 * n)
        upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (n + 1))
    err_lower = n - lower
    err_upper = upper - n
    return err_lower, err_upper



@jit
def test(i, vec):
    mu = 0
    sigma = 1
    p = gaussDistribution(vec, mu + i, sigma)

    s = jnp.sum(-jnp.log(p))
    return s

if __name__ == '__main__':
    n = 100000

    vec = jax.random.uniform(jax.random.key(0), (n,), minval=0, maxval=1)
    test(0, vec)
    import time
    arr = jnp.array([i for i in range(1000)])

    t = time.time()
    v = jax.vmap(lambda i: test(i, vec))
    jax.block_until_ready(v(vec))
    print(time.time() - t)
