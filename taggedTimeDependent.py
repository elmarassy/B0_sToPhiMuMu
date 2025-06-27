import functools
import pstats
import time
import numpy as np
from iminuit import Minuit
import cProfile
import generation
from torchquad import Simpson, MonteCarlo, set_up_backend
import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import plot

set_up_backend("jax", data_type="float64")
integrator = Simpson()


@jit
def individualTimeDependent(normalization, sign, cosThetaL, cosThetaK, phi, t, x, y,
                            K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                            W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                            H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                            Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9):

    cosThetaL2 = cosThetaL * cosThetaL
    cosThetaK2 = cosThetaK * cosThetaK
    cos2ThetaL = 2.0 * cosThetaL2 - 1.0
    sinThetaK2 = 1.0 - cosThetaK2
    sinThetaL2 = 1.0 - cosThetaL2
    sinThetaL = jnp.sqrt(sinThetaL2)
    sinThetaK = jnp.sqrt(sinThetaK2)
    sin2ThetaL = 2.0 * sinThetaL * cosThetaL
    sin2ThetaK = 2.0 * sinThetaK * cosThetaK

    timeDependentTerm = lambda coshFactor, cosFactor, H_i, Z_i: (
            coshFactor*jnp.cosh(y*t) - H_i*jnp.sinh(y*t) + sign*(cosFactor*jnp.cos(x*t) - Z_i*jnp.sin(x*t)))

    return (9.0 / 64.0 / jnp.pi / normalization) * jnp.exp(-t) * (
            timeDependentTerm(K1s, W1s, H1s, Z1s) * sinThetaK2 +
            timeDependentTerm(K1c, W1c, H1c, Z1c) * cosThetaK2 +
            timeDependentTerm(K2s, W2s, H2s, Z2s) * sinThetaK2 * cos2ThetaL +
            timeDependentTerm(K2c, W2c, H2c, Z2c) * cosThetaK2 * cos2ThetaL +
            timeDependentTerm(K3, W3, H3, Z3) * sinThetaK2 * cosThetaK2 * jnp.cos(2 * phi) +
            timeDependentTerm(K4, W4, H4, Z4) * sin2ThetaK * sin2ThetaL * jnp.cos(phi) +
            timeDependentTerm(W5, K5, H5, Z5) * sin2ThetaK * sinThetaL * jnp.cos(phi) +
            timeDependentTerm(W6s, K6s, H6s, Z6s) * sinThetaK2 * cosThetaL +
            timeDependentTerm(K7, W7, H7, Z7) * sin2ThetaK * sinThetaL * jnp.sin(phi) +
            timeDependentTerm(W8, K8, H8, Z8) * sin2ThetaK * sin2ThetaL * jnp.sin(phi) +
            timeDependentTerm(W9, K9, H9, Z9) * sinThetaK2 * sinThetaL2 * jnp.sin(2 * phi)
    )
@jit
def projectCosThetaL(sign, params, cosThetaL):
    norm = integrateIndividual(sign, *params)
    evaluator = jax.vmap(lambda cosThetaL_: integrator.integrate(lambda variables:
                                                                  individualTimeDependent(norm, sign, cosThetaL_, variables[:, 0], variables[:, 1], variables[:, 2], *params),
                                                                  dim=3, N=999, integration_domain=[[-1, 1], [-jnp.pi, jnp.pi], [0, 10]]))
    return evaluator(cosThetaL)

@jit
def projectCosThetaK(sign, params, cosThetaK):
    norm = integrateIndividual(sign, *params)
    evaluator = jax.vmap(lambda cosThetaK_: integrator.integrate(lambda variables:
                                                                 individualTimeDependent(norm, sign, variables[:, 0], cosThetaK_, variables[:, 1], variables[:, 2], *params),
                                                                 dim=3, N=1000, integration_domain=[[-1, 1], [-jnp.pi, jnp.pi], [0, 10]]))
    return evaluator(cosThetaK)

@jit
def projectPhi(sign, params, phi):
    norm = integrateIndividual(sign, *params)
    evaluator = jax.vmap(lambda phi_: integrator.integrate(lambda variables:
                                                                 individualTimeDependent(norm, sign, variables[:, 0], variables[:, 1], phi_, variables[:, 2], *params),
                                                                 dim=3, N=1000, integration_domain=[[-1, 1], [-1, 1], [0, 10]]))
    return evaluator(phi)

@jit
def projectT(sign, params, t):
    norm = integrateIndividual(sign, *params)
    evaluator = jax.vmap(lambda t_: integrator.integrate(lambda variables:
                                                                 individualTimeDependent(norm, sign, variables[:, 0], variables[:, 1], variables[:, 2], t_, *params),
                                                                 dim=3, N=1000, integration_domain=[[-1, 1], [-1, 1], [-jnp.pi, jnp.pi]]))
    return evaluator(t)

@jit
def integrateIndividual(sign, x, y,
                        K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                        W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                        H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                        Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9):
    return (1./8)*(
            (-3*K1c - 6*K1s + K2c + 2*K2s + (3*H1c + 6*H1s - H2c - 2*H2s)*y)/(-1 + y*y) +
            sign * (3*W1c + 6*W1s - W2c - 2*W2s - (3*Z1c + 6*Z1s - Z2c - 2*Z2s)*x)/(1 + x*x)
    )


@jit
def neg_log_likelihood(params, cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS, normB, normBBar):
    pdf_vals = vmap(lambda ctl, ctk, p, t, qSS_, qOS_: (
            (1 + qSS_ * (1 - 2 * wSS)) *
            (1 + qOS_ * (1 - 2 * wOS)) *
            individualTimeDependent(normB, 1, ctl, ctk, p, t, *params) +
            (1 - qSS_ * (1 - 2 * wSS)) *
            (1 - qOS_ * (1 - 2 * wOS)) *
            individualTimeDependent(normBBar, -1, ctl, ctk, p, t, *params)
    ))(cosThetaL, cosThetaK, phi, time, qSS, qOS)
    return -jnp.sum(jnp.log(pdf_vals))


@jit
def helper2(x, y,
            K1c, K3, K4, K5, K6s, K7, K8, K9,
            W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
            H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
            Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9):

    K1s = (3.0/4.0) * (1 - y*y - K1c + y*H1c) + y*H1s
    K2s = (1.0/4.0) * (1 - y*y - K1c + y*H1c) + y*H1s/3.0

    W2s = W1s/3.0
    H2s = H1s/3.0
    Z2s = Z1s/3.0

    K2c = -K1c
    W2c = -W1c
    H2c = -H1c
    Z2c = -Z1c

    normB = integrateIndividual(1, x, y,
                                K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                                W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                                H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                                Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9)
    normBBar = integrateIndividual(-1, x, y,
                                   K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                                   W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                                   H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                                   Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9)
    return normB, normBBar, jnp.array([x, y,
                                       K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                                       W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                                       H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                                       Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9])

def masslessNLLHelper(x, y,
                      K1c, K3, K4, K5, K6s, K7, K8, K9,
                      W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                      H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                      Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9):

    normB, normBBar, arr = helper2(x, y,
                                   K1c, K3, K4, K5, K6s, K7, K8, K9,
                                   W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                                   H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                                   Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9)

    return lambda cosThetaL, cosThetaK, phi, t, qSS, qOS, wSS, wOS: (
        neg_log_likelihood(arr, cosThetaL, cosThetaK, phi, t, qSS, qOS, wSS, wOS, normB, normBBar))


class FullNLL:
    def __init__(self, cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS):
        self.cosThetaL = cosThetaL
        self.cosThetaK = cosThetaK
        self.phi = phi
        self.time = time
        self.qSS = qSS
        self.qOS = qOS
        self.wSS = wSS
        self.wOS = wOS

    def __call__(self, x, y,
                 K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                 W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                 H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                 Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9):

        K1s = (4*(1-y*y) - 3*K1c + K2c + 2*K2s + y*(3*H1c + 6*H1s - H2c - 2*H2s))/6

        normB = integrateIndividual(1, x, y,
                                    K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                                    W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                                    H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                                    Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9)
        normBBar = integrateIndividual(-1, x, y,
                                       K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                                       W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                                       H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                                       Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9)

        likelihood = neg_log_likelihood(jnp.array([x, y,
                                                   K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                                                   W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                                                   H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                                                   Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9]),
                                        self.cosThetaL, self.cosThetaK, self.phi, self.time, self.qSS, self.qOS, self.wSS, self.wOS, normB, normBBar)

        if (jnp.isfinite(likelihood)):
            return float(likelihood)
        return 1e12

@jit
def value(params, cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS):
    return masslessNLLHelper(*params)(cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS)


gradHelper = jax.grad(value, argnums=0)

class MasslessNLL:

    def __init__(self, cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS):
        self.cosThetaL = cosThetaL
        self.cosThetaK = cosThetaK
        self.phi = phi
        self.time = time
        self.qSS = qSS
        self.qOS = qOS
        self.wSS = wSS
        self.wOS = wOS
        self.gradientHelper = gradHelper

    def __call__(self, x, y,
                 K1c, K3, K4, K5, K6s, K7, K8, K9,
                 W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                 H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                 Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9):

        likelihood = masslessNLLHelper(x, y,
                                       K1c, K3, K4, K5, K6s, K7, K8, K9,
                                       W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                                       H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                                       Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9)(self.cosThetaL, self.cosThetaK, self.phi, self.time, self.qSS, self.qOS, self.wSS, self.wOS)
        if jnp.isfinite(likelihood):
            return float(likelihood)
        return 1e12

    @property
    def grad(self):
        return lambda *params: self.gradientHelper(params, self.cosThetaL, self.cosThetaK, self.phi, self.time, self.qSS, self.qOS, self.wSS, self.wOS)


@functools.partial(jit, static_argnums=(1,))
def generateData(key, N, params, B0proportion, effSS, effOS, mistagSS, mistagOS):
    keys = random.split(key, 11)
    mul = 100
    trueTag = random.uniform(keys[0], shape=(mul*N,)) < B0proportion
    trueTagValue = jnp.where(trueTag, 1, -1)

    taggedOS = random.uniform(keys[1], shape=(mul*N,)) < effOS
    taggedSS = random.uniform(keys[2], shape=(mul*N,)) < effSS
    mistaggedOS = random.uniform(keys[3], shape=(mul*N,)) < mistagOS
    mistaggedSS = random.uniform(keys[4], shape=(mul*N,)) < mistagSS

    tagChoiceOS = jnp.where(taggedOS, jnp.where(mistaggedOS, -trueTagValue, trueTagValue), 0)
    tagChoiceSS = jnp.where(taggedSS, jnp.where(mistaggedSS, -trueTagValue, trueTagValue), 0)

    cosThetaL = random.uniform(keys[5], shape=(mul*N,), minval=-1.0, maxval=1.0)
    cosThetaK = random.uniform(keys[6], shape=(mul*N,), minval=-1.0, maxval=1.0)
    phi = random.uniform(keys[7], shape=(mul*N,), minval=-jnp.pi, maxval=jnp.pi)
    t = random.uniform(keys[8], shape=(mul*N,), minval=0.0, maxval=10.0)

    normB = integrateIndividual(1, *params)
    normBBar = integrateIndividual(-1, *params)

    probabilities: jnp.ndarray = jax.vmap(
        lambda flavor, costL, costK, phi_, t_: jax.lax.cond(
            flavor,
            lambda args: individualTimeDependent(normB, 1, *args, *params),
            lambda args: individualTimeDependent(normBBar, -1, *args, *params),
            (costL, costK, phi_, t_)
        )
    )(trueTag, cosThetaL, cosThetaK, phi, t)

    probabilities /= probabilities.max()

    cutoff = random.uniform(keys[9], shape=(mul*N,), minval=0.0, maxval=1.0)

    accept = jnp.nonzero(probabilities > cutoff, size=N, fill_value=-1)[0]
    return cosThetaL[accept], cosThetaK[accept], phi[accept], t[accept], tagChoiceOS[accept], tagChoiceSS[accept]


def run(key, massless, nEvents, nToys, effSS, effOS, wSS, wOS, B0proportion=0.5):

    paramNames = generation.getFitParamNames(massless)

    values = np.empty(shape=(nToys, len(paramNames)))
    errors = np.empty(shape=(nToys, len(paramNames)))
    keys = random.split(key, nToys)

    for i in range(nToys):

        fitParameters = generation.getFitParams(massless)
        tStart = time.time()
        data = generateData(keys[i], nEvents, generation.getAllParams(), B0proportion, effSS, effOS, wSS, wOS)

        if massless:
            nll = MasslessNLL(*data, wSS, wOS)
        else:
            nll = FullNLL(*data, wSS, wOS)

        m = Minuit(nll, *fitParameters)

        m.fixed["x"] = True
        m.fixed["y"] = True

        m.limits = [(-1 - 4*abs(param), 1 + 4*abs(param)) for param in fitParameters]
        m.strategy = 0
        m.errordef = Minuit.LIKELIHOOD
        m.migrad()
        m.hesse()

        if not m.fmin.is_valid:
            print(m.fmin)

        values[i, :] = np.array(m.values)
        errors[i, :] = np.array(m.errors)
        print(f"Performed fit #{i} in {time.time() - tStart:.3f} seconds")

    trueValues = np.array(generation.getFitParams(massless)).T
    pulls = (values - trueValues) / errors

    return paramNames, pulls, values

def main():
    saveProjectionsValue = 5
    massless = True
    nEvents = 4000
    nToys = 250
    names, pulls, values = run(random.key(1), massless, nEvents, nToys, 1, 1, 0.0, 0.0)
    plot.project(saveProjectionsValue, massless, pulls, values)
    plot.plotSummary(names, pulls, True, False)
    plot.plotSummary(names, values, True, True)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('time')
    # stats.print_stats(100)