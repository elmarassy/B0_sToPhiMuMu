import functools
import pstats
import time
import numpy as np
from iminuit import Minuit
import cProfile
import generation
from torchquad import Simpson, set_up_backend
import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.math as tfp_math
import plot



import jax
import jax.numpy as jnp

import tools

#
# @jax.jit
# def wofz(z):
#     return tfp_math.erfcx(-1j*z)

set_up_backend("jax", data_type="float64")
integrator = Simpson()

# @jax.jit
# def faddeeva(x):
#     return tfp_math.erfcx(-1j*x)

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
    # sigmaT = 0.1
    # gamma = 1
    #
    # a = 1/(2*sigmaT**2)
    # b = (t/(sigmaT**2)-gamma)
    # wX = x*gamma
    # wY = y*gamma

    # C = jnp.sqrt(jnp.pi/a) / 4
    # timeDependentTerm = lambda coshFactor, cosFactor, H_i, Z_i: (
    #         coshFactor*jnp.cosh(y*t) - H_i*jnp.sinh(y*t) + sign*(cosFactor*jnp.cos(x*t) - Z_i*jnp.sin(x*t)))
    # xTerm = jnp.sqrt(jnp.pi / a) * wofz((wX - 1j*b) / (2*jnp.sqrt(a))) / 2
    #
    # z1 = 1j*(b + wY) / 2*jnp.sqrt(a)
    # z2 = 1j*(b - wY) / 2*jnp.sqrt(a)
    # f1 = wofz(z1)
    # f2 = wofz(z2)
    # # d = b*wY/(2*a)
    # coshTerm = C * jnp.real(f1 + f2)
    # sinhTerm = C * jnp.real(f1 - f2)
    # # sinhTerm = C * (jnp.sinh(d) * (jnp.exp(d) * jsp.erf(z1) - jnp.exp(-d) * jsp.erf(z2))/2)

    # timeDependentTerm = lambda coshFactor, cosFactor, H_i, Z_i: (
    #         jnp.exp(-a*t*t)*(coshFactor*coshTerm - H_i * sinhTerm + sign*(cosFactor*jnp.real(xTerm) - Z_i*jnp.imag(xTerm))))
    gamma = 1
    sigmaT = 0.01
    a = (1/jnp.sqrt(2)) * (t/sigmaT + gamma*sigmaT*(1j*x-1))
    b = t*gamma*(1j*x-1) + (gamma * sigmaT * (1j*x-1))**2 / 2

    xTerm = 0.5*(1 + tools.erf(a))*jnp.exp(b)

    timeDependentTerm = lambda coshFactor, cosFactor, H_i, Z_i: (
            coshFactor*jnp.cosh(y*t) - H_i*jnp.sinh(y*t) + sign*(cosFactor*jnp.real(xTerm) - Z_i*jnp.imag(xTerm)))

    return (9.0 / 64.0 / jnp.pi / normalization) * jnp.exp(-t) * (
            timeDependentTerm(K1s, W1s, H1s, Z1s) * sinThetaK2 +
            timeDependentTerm(K1c, W1c, H1c, Z1c) * cosThetaK2 +
            timeDependentTerm(K2s, W2s, H2s, Z2s) * sinThetaK2 * cos2ThetaL +
            timeDependentTerm(K2c, W2c, H2c, Z2c) * cosThetaK2 * cos2ThetaL +
            timeDependentTerm(K3, W3, H3, Z3) * sinThetaK2 * sinThetaL2 * jnp.cos(2 * phi) +
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

    return integrator.integrate(lambda variables:
                                individualTimeDependent(1, sign, variables[:, 0], variables[:, 1], variables[:, 2], variables[:, 3], x, y,
                                                        K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                                                        W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                                                        H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                                                        Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9),
                                dim=4, N=200000, integration_domain=[[-1, 1], [-1, 1], [-jnp.pi, jnp.pi], [0.0, 10.0]])
    # return evaluator(t)

    # return (1./8)*(
    #         (-3*K1c - 6*K1s + K2c + 2*K2s + (3*H1c + 6*H1s - H2c - 2*H2s)*y)/(-1 + y*y) +
    #         sign * (3*W1c + 6*W1s - W2c - 2*W2s - (3*Z1c + 6*Z1s - Z2c - 2*Z2s)*x)/(1 + x*x)
    # )


@jit
def neg_log_likelihood(params, cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS, normB, normBBar):
    pdf_vals = vmap(lambda ctl, ctk, p, t, qSS_, qOS_, wSS_, wOS_: (
            (1 + qSS_ * (1 - 2 * wSS_)) *
            (1 + qOS_ * (1 - 2 * wOS_)) *
            individualTimeDependent(normB, 1, ctl, ctk, p, t, *params) +
            (1 - qSS_ * (1 - 2 * wSS_)) *
            (1 - qOS_ * (1 - 2 * wOS_)) *
            individualTimeDependent(normBBar, -1, ctl, ctk, p, t, *params)
    ))(cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS)
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

@jit
def helperACP(x, y,
            K1c, K3, K4, K5, K6s, K7, K8, K9,
            ACP, W1c, W3, W4, W5, W6s, W7, W8, W9,
            H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
            Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9):

    K1s = (3.0/4.0) * (1 - y*y - K1c + y*H1c) + y*H1s
    K2s = (1.0/4.0) * (1 - y*y - K1c + y*H1c) + y*H1s/3.0

    W1s = 3*(4 * ACP * (1 + x*x) + (16*Z1s/3 + 4*Z1c)*x - 4*W1c)/16.0
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

def masslessNLLHelperACP(x, y,
                         K1c, K3, K4, K5, K6s, K7, K8, K9,
                         ACP, W1c, W3, W4, W5, W6s, W7, W8, W9,
                         H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                         Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9):
    normB, normBBar, arr = helperACP(x, y,
                                   K1c, K3, K4, K5, K6s, K7, K8, K9,
                                   ACP, W1c, W3, W4, W5, W6s, W7, W8, W9,
                                   H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                                   Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9)

    return lambda cosThetaL, cosThetaK, phi, t, qSS, qOS, wSS, wOS: (
        neg_log_likelihood(arr, cosThetaL, cosThetaK, phi, t, qSS, qOS, wSS, wOS, normB, normBBar))

@jit
def masslessNLL2(params, data):
    normB, normBBar, arr = helper2(26.93, 0.124, *params)
    return neg_log_likelihood(arr, *data, normB, normBBar)

def masslessNLL3(data):
    @jit
    def fn(params):
        return masslessNLL2(params, data)
    return fn

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
def valueW1s(params, cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS):
    return masslessNLLHelper(*params)(cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS)

@jit
def valueACP(params, cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS):
    return masslessNLLHelperACP(*params)(cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS)


gradHelperW1s = jax.grad(valueW1s, argnums=0)
gradHelperACP = jax.grad(valueACP, argnums=0)
class MasslessNLL:

    def __init__(self, cosThetaL, cosThetaK, phi, time, qSS, qOS, wSS, wOS, useACP):
        self.cosThetaL = cosThetaL
        self.cosThetaK = cosThetaK
        self.phi = phi
        self.time = time
        self.qSS = qSS
        self.qOS = qOS
        self.wSS = wSS
        self.wOS = wOS
        self.useACP = useACP
        self.gradientHelper = gradHelperACP if useACP else gradHelperW1s

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
    return cosThetaL[accept], cosThetaK[accept], phi[accept], t[accept], tagChoiceSS[accept], tagChoiceOS[accept], jnp.full(N, mistagSS), jnp.full(N, mistagOS)


def run(key, massless, useACP, nEvents, nToys, effSS, effOS, wSS, wOS, saveProjectionValue, B0proportion=0.5):

    paramNames = generation.getFitParamNames(massless, useACP)
    trueValues = np.array(generation.getFitParams(massless, useACP))

    savedData = []
    values = np.empty(shape=(nToys, len(paramNames)))
    errors = np.empty(shape=(nToys, len(paramNames)))
    pulls = np.empty(shape=(nToys, len(paramNames)))

    i = 0
    failedCounter = 0
    while i < nToys:
        key, key1 = jax.random.split(key)
        fitParameters = generation.getFitParams(massless, useACP)
        tStart = time.time()
        data = generateData(key1, nEvents, generation.getAllParamsFromMassless(*fitParameters, useACP), B0proportion, effSS, effOS, wSS, wOS)

        if massless:
            nll = MasslessNLL(*data, useACP)
        else:
            nll = FullNLL(*data, useACP)

        m = Minuit(nll, *fitParameters)

        # m.fixed["x"] = True
        # m.fixed["y"] = True
        #
        # m.fixed["W1s"] = True
        # m.fixed["W1c"] = True
        # m.fixed["W2s"] = True
        # m.fixed["W2c"] = True
        # m.fixed["W3"] = True
        # m.fixed["W4"] = True
        # m.fixed["K5"] = True
        # m.fixed["K6s"] = True
        # m.fixed["W7"] = True
        # m.fixed["K8"] = True
        # m.fixed["K9"] = True
        # #
        # m.fixed["Z1s"] = True
        # m.fixed["Z1c"] = True
        # m.fixed["Z2s"] = True
        # m.fixed["Z2c"] = True
        # m.fixed["Z3"] = True
        # m.fixed["Z4"] = True
        # m.fixed["Z5"] = True
        # m.fixed["Z6s"] = True
        # m.fixed["Z7"] = True
        # m.fixed["Z8"] = True
        # m.fixed["Z9"] = True

        m.limits = [(-1 - 4*abs(param), 1 + 4*abs(param)) for param in fitParameters]
        m.strategy = 0
        m.errordef = Minuit.LIKELIHOOD
        m.migrad()
        m.hesse()
        print('numnber: ',m.nfcn)
        if not m.fmin.is_valid or not m.fmin.has_accurate_covar or m.fmin.has_parameters_at_limit or m.fmin.hesse_failed:
            print(m.fmin)
            print(f"Failed fit #{i} in {time.time() - tStart:.3f} seconds")
            failedCounter += 1
            continue
        vals = np.array(m.values)
        errs = np.array(m.errors)
        values[i, :] = vals
        errors[i, :] = errs
        pulls[i, :] = (vals - trueValues) / errs
        if np.any(np.abs(pulls[i, :]) > saveProjectionValue):
            savedData.append((data, m.covariance.correlation()))

        print(f"Performed fit #{i} in {time.time() - tStart:.3f} seconds")
        i += 1

    return paramNames, pulls, values, savedData

def main():
    saveProjectionsValue = 4
    massless = True
    useACP = False
    nEvents = 4000
    nToys = 3
    # wSS = 0.42
    # wOS = 0.39
    # effSS = 0.40
    # effOS = 0.80
    wSS = 0.0
    wOS = 0.0
    effSS = 1
    effOS = 1
    # names, pulls, values, savedData = run(random.key(1), massless, useACP, nEvents, nToys, effSS, effOS, wSS, wOS, saveProjectionsValue)
    plot.project(saveProjectionsValue, [], [], [], effSS, effOS, wSS, wOS, "plots", useACP)

    # plot.project(saveProjectionsValue, pulls, values, savedData, effSS, effOS, wSS, wOS, "plots", useACP)
    # plot.plotSummary(names, pulls, True, False, "plots")
    # plot.plotSummary(names, values, True, True, "plots", generation.getFitParams(massless, useACP))


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('time')
    # stats.print_stats(100)