import os

from matplotlib.offsetbox import AnchoredText

os.environ["JAX_ENABLE_X64"] = "True"
import functools
import time
import numpy as np
from iminuit import Minuit
import generation as generation
from jax import jit, random
import jax
import jax.numpy as jnp
import tools


cosThetaLRange = [-1.0, 1.0]
cosThetaKRange = [-1.0, 1.0]
phiRange = [-jnp.pi, jnp.pi]
timeRange = [0.0, 7.0]
massRange = [5.3, 5.7]

x = generation.trueValues['$x$']
y = generation.trueValues['$y$']
gamma = generation.trueValues['$gamma$']
acceptance = lambda t: 1.05 / (1+13.3*jnp.exp(-8.1*t))
coshIntegral, sinhIntegral, cosIntegral, sinIntegral = tools.getIntegrals(acceptance, timeRange, gamma, x, y)


@jit
def individualTimeDependent(normalization, sign, cosThetaK, cosThetaL, phi, t, mass,
                            K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                            W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                            H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                            Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):

    cosThetaL2 = cosThetaL * cosThetaL
    cosThetaK2 = cosThetaK * cosThetaK
    cos2ThetaL = 2.0 * cosThetaL2 - 1.0
    sinThetaK2 = 1.0 - cosThetaK2
    sinThetaL2 = 1.0 - cosThetaL2
    sinThetaL = jnp.sqrt(sinThetaL2)
    sinThetaK = jnp.sqrt(sinThetaK2)
    sin2ThetaL = 2.0 * sinThetaL * cosThetaL
    sin2ThetaK = 2.0 * sinThetaK * cosThetaK

    osc = tools.convolutionIntegral(gamma, 1j*x, t)
    decayPlus = tools.convolutionIntegral(gamma, y, t)
    decayMinus = tools.convolutionIntegral(gamma, -y, t)

    def helper(coshFactor, cosFactor, H_i, Z_i):
        return coshFactor*(decayPlus + decayMinus)/2. - H_i*(decayPlus - decayMinus)/2. + sign*(cosFactor*jnp.real(osc) - Z_i*jnp.imag(osc))

    return (9.0 / 64.0 / jnp.pi / normalization) * gamma * (1-y*y) * acceptance(t) * (
            helper(K1s, W1s, H1s, Z1s) * sinThetaK2 +
            helper(K1c, W1c, H1c, Z1c) * cosThetaK2 +
            helper(K2s, W2s, H2s, Z2s) * sinThetaK2 * cos2ThetaL +
            helper(K2c, W2c, H2c, Z2c) * cosThetaK2 * cos2ThetaL +
            helper(K3, W3, H3, Z3) * sinThetaK2 * sinThetaL2 * jnp.cos(2 * phi) +
            helper(K4, W4, H4, Z4) * sin2ThetaK * sin2ThetaL * jnp.cos(phi) +
            helper(W5, K5, H5, Z5) * sin2ThetaK * sinThetaL * jnp.cos(phi) +
            helper(W6s, K6s, H6s, Z6s) * sinThetaK2 * cosThetaL +
            helper(K7, W7, H7, Z7) * sin2ThetaK * sinThetaL * jnp.sin(phi) +
            helper(W8, K8, H8, Z8) * sin2ThetaK * sin2ThetaL * jnp.sin(phi) +
            helper(W9, K9, H9, Z9) * sinThetaK2 * sinThetaL2 * jnp.sin(2 * phi)
    ) * tools.gaussDistribution(mass, m, sigmaM)


@jit
def integrateIndividual(sign,
                        K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                        W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                        H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                        Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):

# @jit
# def integrateIndividual(sign,
#                         A1s, A1c, A2s, A2c, K3, K4, K5, K6s, K7, K8, K9,
#                         B1s, B1c, B2s, B2c, W3, W4, W5, W6s, W7, W8, W9,
#                         H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
#                         Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):
#     K1c = A1c + B1c
#     W1c = B1c - A1c
#     K1s = A1s + B1s
#     W1s = B1s - A1s
#     K2s = A2s + B2s
#     W2s = B2s - A2s
#     K2c = A2c + B2c
#     W2c = B2c - A2c

    return (1./8) * (1-y*y) * gamma * (
            (3*W1c + 6*W1s - W2c - 2*W2s) * cosIntegral * sign +
            (3*K1c + 6*K1s - K2c - 2*K2s) * coshIntegral -
            (3*Z1c + 6*Z1s - Z2c - 2*Z2s) * sinIntegral * sign -
            (3*H1c + 6*H1s - H2c - 2*H2s) * sinhIntegral
    )


@jit
def projectSignalCosThetaL(cosThetaL, sign,
                     K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                     W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                     H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                     Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):

    norm = integrateIndividual(sign,
                               K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                               W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                               H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                               Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)

    return (3.0/16.0/norm) * gamma * (1 - y*y) * ((W1c + 2*W1s - (1 - 2*cosThetaL**2)*(W2c + 2*W2s) + 2*K6s*cosThetaL)*cosIntegral*sign +
                                           (K1c + 2*K1s - (1 - 2*cosThetaL**2)*(K2c + 2*K2s) + 2*W6s*cosThetaL)*coshIntegral -
                                           (Z1c + 2*Z1s - Z2c - 2*Z2s + 2*(Z2c + 2*Z2s)*cosThetaL**2 + 2*Z6s*cosThetaL)*sinIntegral*sign -
                                           (H1c + 2*H1s - H2c - 2*H2s + 2*(H2c + 2*H2s)*cosThetaL**2 + 2*H6s*cosThetaL)*sinhIntegral)

@jit
def projectSignalCosThetaK(cosThetaK, sign,
                     K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                     W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                     H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                     Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):

    norm = integrateIndividual(sign,
                               K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                               W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                               H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                               Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)

    # f = lambda ctk, values: individualTimeDependent(norm, sign, values[:, 0], ctk, values[:, 1], values[:, 2], values[:, 3],
    #                                                      K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
    #                                                      W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
    #                                                      H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
    #                                                      Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)
    #
    # return jnp.array([tools.integrator.integrate(lambda values: f(ctk, values), dim=4, N=500000, integration_domain=[cosThetaLRange, phiRange, timeRange, massRange]) for ctk in cosThetaK])

    return -(3.0/16.0/norm)*gamma*(1-y*y)*(
            (W2s - 3*W1s + (-3*W1c + 3*W1s + W2c - W2s)*cosThetaK**2)*cosIntegral*sign +
            (K2s - 3*K1s + (-3*K1c + 3*K1s + K2c - K2s)*cosThetaK**2)*coshIntegral -
            (Z2s - 3*Z1s + (-3*Z1c + 3*Z1s + Z2c - Z2s)*cosThetaK**2)*sinIntegral*sign -
            (H2s - 3*H1s + (-3*H1c + 3*H1s + H2c - H2s)*cosThetaK**2)*sinhIntegral)


@jit
def projectSignalPhi(phi, sign,
               K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
               W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
               H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
               Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):

    norm = integrateIndividual(sign,
                               K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                               W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                               H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                               Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)

    return (1./(16*jnp.pi*norm)) * gamma * (1-y*y) * (
        (3*K1c + 6*K1s - K2c - 2*K2s)*coshIntegral - (3*H1c + 6*H1s - H2c - 2*H2s)*sinhIntegral +
        sign*((3*W1c + 6*W1s - W2c - 2*W2s)*cosIntegral - (3*Z1c + 6*Z1s - Z2c - 2*Z2s)*sinIntegral) +
        4*(K3*coshIntegral - H3*sinhIntegral + sign*(W3*cosIntegral - Z3*sinIntegral))*jnp.cos(2*phi) +
        4*(K9*coshIntegral - H9*sinhIntegral + sign*(W9*cosIntegral - Z9*sinIntegral))*jnp.sin(2*phi)
    )


@jit
def projectSignalT(t, sign,
             K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
             W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
             H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
             Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):

    norm = integrateIndividual(sign,
                               K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                               W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                               H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                               Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)

    osc = tools.convolutionIntegral(gamma, 1j*x, t)
    decayPlus = tools.convolutionIntegral(gamma, y, t)
    decayMinus = tools.convolutionIntegral(gamma, -y, t)

    return (1./8.) * (1-y*y) * gamma * acceptance(t) * (
            (3*K1c + 6*K1s - K2c - 2*K2s) * (decayPlus + decayMinus)/2 -
            (3*H1c + 6*H1s - H2c - 2*H2s) * (decayPlus - decayMinus)/2 +
            sign * (
                (3*W1c + 6*W1s - W2c - 2*W2s) * jnp.real(osc) -
                (3*Z1c + 6*Z1s - Z2c - 2*Z2s) * jnp.imag(osc)
            )
    ) / norm


@jit
def projectSignalMass(mass, sign,
                      K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                      W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                      H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                      Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):
    return tools.gaussDistribution(mass, m, sigmaM)


@jit
def projectBackgroundAngles(angle, *backgroundParams):
    return jnp.full_like(angle, 1/(angle[-1] - angle[0]))


@jit
def projectBackgroundTime(t, c0, c1, c2, kM):
    return (acceptance(t) * (c0*tools.exponentialDistribution(t, c1, timeRange) + (1-c0)*tools.exponentialDistribution(t, c2, timeRange))) / integralBackground(c0, c1, c2, kM)


@jit
def projectBackgroundMass(mass, c0, c1, c2, kM):
    return tools.exponentialDistribution(mass, kM, massRange)


@jit
def backgroundDistribution(time, mass, c0, c1, c2, kM):
    timeDistr = c0*tools.exponentialDistribution(time, c1, timeRange) + (1-c0)*tools.exponentialDistribution(time, c2, timeRange)
    massDistr = tools.exponentialDistribution(mass, kM, massRange)
    normalization = 0.5 * 0.5 * 1/(2*jnp.pi)
    return timeDistr * massDistr * normalization * acceptance(time)


@jit
def integralBackground(c0, c1, c2, kM):
    return tools.integrator.integrate(lambda var: (acceptance(var[:, 0]) * (c0*tools.exponentialDistribution(var[:, 0], c1, timeRange) + (1-c0)*tools.exponentialDistribution(var[:, 0], c2, timeRange))), dim=1, N=9999, integration_domain=[timeRange])


@jit
def likelihood2(signalParams, backgroundParams, f, cosThetaK, cosThetaL, phi, time, mass, qSS, qOS, wSS, wOS, normB, normBBar):
    background = backgroundDistribution(time, mass, *backgroundParams) / integralBackground(*backgroundParams)
    weightB = (1 + qSS * (1 - 2 * wSS)) * (1 + qOS * (1 - 2 * wOS)) / 4
    weightBBar = (1 - qSS * (1 - 2 * wSS)) * (1 - qOS * (1 - 2 * wOS)) / 4
    b = individualTimeDependent(normB, 1, cosThetaK, cosThetaL, phi, time, mass, *signalParams)
    bBar = individualTimeDependent(normBBar, -1, cosThetaK, cosThetaL, phi, time, mass, *signalParams)
    pdf = (
            weightB *
            (f*b + (1-f)*background) +
            weightBBar *
            (f*bBar + (1-f)*background))
    return pdf

@jit
def likelihood(signalParams, backgroundParams, f, cosThetaK, cosThetaL, phi, time, mass, qSS, qOS, wSS, wOS, normB, normBBar):
    return -jnp.sum(jnp.log(likelihood2(signalParams, backgroundParams, f, cosThetaK, cosThetaL, phi, time, mass, qSS, qOS, wSS, wOS, normB, normBBar)))

@jit
def phiLikelihood(signalParams, f, phi, qSS, qOS, wSS, wOS, normB, normBBar):
    background = 1/(2*jnp.pi)
    weightB = (1 + qSS * (1 - 2 * wSS)) * (1 + qOS * (1 - 2 * wOS)) / 4
    weightBBar = (1 - qSS * (1 - 2 * wSS)) * (1 - qOS * (1 - 2 * wOS)) / 4
    b = projectSignalPhi(phi, 1, *signalParams) / normB
    bBar = projectSignalPhi(phi, -1, *signalParams) / normBBar
    pdf = (
            weightB *
            (f*b + (1-f)*background) +
            weightBBar *
            (f*bBar + (1-f)*background))
    return -jnp.sum(jnp.log(pdf))


@jit
def helper2(K1c, K3, K4, K5, K6s, K7, K8, K9,
            W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
            H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
            Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):

    # K1s = (3.0/4.0) * (1 - y*y - K1c + y*H1c) + y*H1s
    # K2s = (1.0/4.0) * (1 - y*y - K1c + y*H1c) + y*H1s/3.0
    # (K1c, K3, K4, K5, K6s, K7, K8, K9,
    # W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
    # H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
    # Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM) = generation.transformBack(K1c, K3, K4, K5, K6s, K7, K8, K9,
    #                                                                              W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
    #                                                                              H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
    #                                                                              Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)

    K1s = (3.0/4.0)*(1 - K1c + y*(4.0*H1s/3.0 + H1c))
    K2s = K1s / 3.0


    W2s = W1s/3.0
    H2s = H1s/3.0
    Z2s = Z1s/3.0

    K2c = -K1c
    W2c = -W1c
    H2c = -H1c
    Z2c = -Z1c


    normB = integrateIndividual(1,
                                K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                                W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                                H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                                Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)
    normBBar = integrateIndividual(-1,
                                   K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                                   W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                                   H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                                   Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)
    return normB, normBBar, jnp.array([K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
                                       W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
                                       H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
                                       Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM])

def masslessNLLHelper(K1c, K3, K4, K5, K6s, K7, K8, K9,
                      W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                      H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                      Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM,
                      c0, c1, c2, kM, f):

    normB, normBBar, arr = helper2(K1c, K3, K4, K5, K6s, K7, K8, K9,
                                   W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                                   H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                                   Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)
    return lambda cosThetaK, cosThetaL, phi, t, mass, qSS, qOS, wSS, wOS: (
        likelihood(arr, (c0, c1, c2, kM), f, cosThetaK, cosThetaL, phi, t, mass, qSS, qOS, wSS, wOS, normB, normBBar))

def phiHelper(K1c, K3, K4, K5, K6s, K7, K8, K9,
              W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
              H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
              Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM,
              c0, c1, c2, kM, f):
    normB, normBBar, arr = helper2(K1c, K3, K4, K5, K6s, K7, K8, K9,
                                                             W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                                                             H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                                                             Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)
    return lambda phi, qSS, qOS, wSS, wOS: (
        phiLikelihood(arr, f, phi, qSS, qOS, wSS, wOS, normB, normBBar))

@jit
def value(params, cosThetaK, cosThetaL, phi, time, mass, qSS, qOS, wSS, wOS):
    return masslessNLLHelper(*params)(cosThetaK, cosThetaL, phi, time, mass, qSS, qOS, wSS, wOS)

gradHelper = jax.grad(value, argnums=0)

@jit
def phi(params, phi, qSS, qOS, wSS, wOS):
    return phiHelper(*params)(phi, qSS, qOS, wSS, wOS)
phiGradHelper = jax.grad(phi, argnums=0)
class phiNLL:

    def __init__(self, phi, qSS, qOS, wSS, wOS):
        self.phi = phi
        self.qSS = qSS
        self.qOS = qOS
        self.wSS = wSS
        self.wOS = wOS
        self.gradientHelper = phiGradHelper

    def __call__(self, K1c, K3, K4, K5, K6s, K7, K8, K9,
                 W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                 H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                 Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM,
                 c0, c1, c2, kM, f):
        likelihood = phiHelper(K1c, K3, K4, K5, K6s, K7, K8, K9,
                                      W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                                      H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                                      Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM,
                                      c0, c1, c2, kM, f)(self.phi, self.qSS, self.qOS, self.wSS, self.wOS)
        if jnp.isfinite(likelihood):
            return float(likelihood)
        return 1e12
    @property
    def grad(self):
        return lambda *params: self.gradientHelper(params, self.phi, self.qSS, self.qOS, self.wSS, self.wOS)

class MasslessNLL:

    def __init__(self, cosThetaK, cosThetaL, phi, time, mass, qSS, qOS, wSS, wOS):
        self.cosThetaL = cosThetaL
        self.cosThetaK = cosThetaK
        self.phi = phi
        self.time = time
        self.mass = mass
        self.qSS = qSS
        self.qOS = qOS
        self.wSS = wSS
        self.wOS = wOS
        self.gradientHelper = gradHelper

    def __call__(self,
                 K1c, K3, K4, K5, K6s, K7, K8, K9,
                 W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                 H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                 Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM,
                 c0, c1, c2, kM, f):

        likelihood = masslessNLLHelper(K1c, K3, K4, K5, K6s, K7, K8, K9,
                                       W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                                       H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                                       Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM,
                                       c0, c1, c2, kM, f)(self.cosThetaK, self.cosThetaL, self.phi, self.time, self.mass, self.qSS, self.qOS, self.wSS, self.wOS)
        if jnp.isfinite(likelihood):
            return float(likelihood)
        return 1e12

    @property
    def grad(self):
        return lambda *params: self.gradientHelper(params, self.cosThetaK, self.cosThetaL, self.phi, self.time, self.mass, self.qSS, self.qOS, self.wSS, self.wOS)

@functools.partial(jit, static_argnums=(1,))
def generate(key, N, signalParams, backgroundParams, f, B0proportion, effSS, effOS, mistagSS, mistagOS):
    keys = random.split(key, 17)
    mul = 100

    signal = random.uniform(keys[0], shape=(N,)) < f
    trueTag = random.uniform(keys[1], shape=(mul*N,)) < B0proportion
    trueTagValue = jnp.where(trueTag, 1, -1)

    taggedOS = random.uniform(keys[2], shape=(mul*N,)) < effOS
    taggedSS = random.uniform(keys[3], shape=(mul*N,)) < effSS
    mistaggedOS = random.uniform(keys[4], shape=(mul*N,)) < mistagOS
    mistaggedSS = random.uniform(keys[5], shape=(mul*N,)) < mistagSS

    tagChoiceOS = jnp.where(taggedOS, jnp.where(mistaggedOS, -trueTagValue, trueTagValue), 0)
    tagChoiceSS = jnp.where(taggedSS, jnp.where(mistaggedSS, -trueTagValue, trueTagValue), 0)

    cosThetaLSignal = random.uniform(keys[6], shape=(mul*N,), minval=-1.0, maxval=1.0)
    cosThetaKSignal = random.uniform(keys[7], shape=(mul*N,), minval=-1.0, maxval=1.0)
    phiSignal = random.uniform(keys[8], shape=(mul*N,), minval=-jnp.pi, maxval=jnp.pi)
    tSignal = random.uniform(keys[9], shape=(mul*N,), minval=timeRange[0], maxval=timeRange[1])
    massSignal = random.normal(keys[10], shape=(mul*N,)) * 5*signalParams[-1] + signalParams[-2]
    # massSignal = random.uniform(keys[10], shape=(mul*N,), minval=massRange[0], maxval=massRange[1])
    normB = integrateIndividual(1, *signalParams)
    normBBar = integrateIndividual(-1, *signalParams)
    norm = jnp.where(trueTag, normB, normBBar)
    signalProbabilities = individualTimeDependent(norm, trueTagValue, cosThetaKSignal, cosThetaLSignal, phiSignal, tSignal, massSignal, *signalParams)
    signalProbabilities /= signalProbabilities.max()

    cosThetaLBackground = random.uniform(keys[11], shape=(mul*N,), minval=-1.0, maxval=1.0)
    cosThetaKBackground = random.uniform(keys[12], shape=(mul*N,), minval=-1.0, maxval=1.0)
    phiBackground = random.uniform(keys[13], shape=(mul*N,), minval=-jnp.pi, maxval=jnp.pi)
    tBackground = random.uniform(keys[14], shape=(mul*N,), minval=timeRange[0], maxval=timeRange[1])
    massBackground = random.uniform(keys[15], shape=(mul*N,), minval=massRange[0], maxval=massRange[1])
    backgroundProbabilities = backgroundDistribution(tBackground, massBackground, *backgroundParams)
    backgroundProbabilities /= backgroundProbabilities.max()

    cutoff = random.uniform(keys[16], shape=(mul*N,), minval=0.0, maxval=1.0)
    signalAccept = jnp.nonzero(signalProbabilities > cutoff, size=N, fill_value=-1)[0]
    backgroundAccept = jnp.nonzero(backgroundProbabilities > cutoff, size=N, fill_value=-1)[0]

    cosThetaLSignal = cosThetaLSignal[signalAccept]
    cosThetaKSignal = cosThetaKSignal[signalAccept]
    phiSignal = phiSignal[signalAccept]
    tSignal = tSignal[signalAccept]
    massSignal = massSignal[signalAccept]
    tagChoiceSSSignal = tagChoiceSS[signalAccept]
    tagChoiceOSSignal = tagChoiceOS[signalAccept]

    cosThetaLBackground = cosThetaLBackground[backgroundAccept]
    cosThetaKBackground = cosThetaKBackground[backgroundAccept]
    phiBackground = phiBackground[backgroundAccept]
    tBackground = tBackground[backgroundAccept]
    massBackground = massBackground[backgroundAccept]
    tagChoiceSSBackground = tagChoiceSS[backgroundAccept]
    tagChoiceOSBackground = tagChoiceOS[backgroundAccept]

    cosThetaL = jnp.where(signal, cosThetaLSignal, cosThetaLBackground)
    cosThetaK = jnp.where(signal, cosThetaKSignal, cosThetaKBackground)
    phi = jnp.where(signal, phiSignal, phiBackground)
    t = jnp.where(signal, tSignal, tBackground)
    mass = jnp.where(signal, massSignal, massBackground)
    tagChoiceSS = jnp.where(signal, tagChoiceSSSignal, tagChoiceSSBackground)
    tagChoiceOS = jnp.where(signal, tagChoiceOSSignal, tagChoiceOSBackground)

    return cosThetaK, cosThetaL, phi, t, mass, tagChoiceSS, tagChoiceOS, jnp.full(N, mistagSS), jnp.full(N, mistagOS)

import uncertainties
from uncertainties import umath

def run(nEvents, nToys, effSS, effOS, wSS, wOS, tagged, q2_range, generationName, B0proportion=0.5):
    key = jax.random.key(2)
    if not tagged:
        effSS = 0
        effOS = 0
    paramNames = generation.getFitParamNames(True)
    pars = generation.getFitParams(True, q2_range, generationName)
    pars = (list(generation.transform(*pars[0])), pars[1], pars[2])
    trueValues = np.array(pars[0] + pars[1] + [pars[2]])
    savedData = []
    deltavals = []
    deltaerrs = []
    indexH3 = paramNames.index('$H_3$')
    indexH9 = paramNames.index('$H_9$')
    indexK3 = paramNames.index('$K_3$')
    indexK9 = paramNames.index('$K_9$')
    indices = [indexH3, indexH9, indexK3, indexK9]
    H3 = trueValues[indexH3]
    H9 = trueValues[indexH9]
    K3 = trueValues[indexK3]
    K9 = trueValues[indexK9]
    A = K9*float(coshIntegral) - H9*float(sinhIntegral)
    B = K3*float(coshIntegral) - H3*float(sinhIntegral)
    import math
    trueDelta = math.atan((B/A))

    values = np.empty(shape=(nToys, len(paramNames)))
    errors = np.empty(shape=(nToys, len(paramNames)))
    pulls = np.empty(shape=(nToys, len(paramNames)))

    i = 0
    failedCounter = 0
    while i < nToys:
        key, key1 = jax.random.split(key)
        tStart = time.time()
        pars = generation.getFitParams(True, q2_range, generationName)
        pars = (list(generation.transform(*pars[0])), pars[1], pars[2])

        data = generate(key1, nEvents, generation.getAllSignalParamsFromMassless(*pars[0]), pars[1], pars[2], B0proportion, effSS, effOS, wSS, wOS)
        nll = MasslessNLL(*data)
        m = Minuit(nll, *pars[0], *pars[1], pars[2])

        limits = [(-1 - 4*abs(param), 1 + 4*abs(param)) for param in trueValues]
        limits[generation.getFitParamNames(True).index("$f$")] = (0, 1)
        limits[generation.getFitParamNames(True).index("$c_0$")] = (0, 1)
        if not tagged:
            m.fixed['W1s'] = True
            m.fixed['W1c'] = True
            m.fixed['W3'] = True
            m.fixed['W4'] = True
            m.fixed['K5'] = True
            m.fixed['K6s'] = True
            m.fixed['W7'] = True
            m.fixed['K8'] = True
            m.fixed['K9'] = True
            m.fixed['Z1s'] = True
            m.fixed['Z1c'] = True
            m.fixed['Z3'] = True
            m.fixed['Z4'] = True
            m.fixed['Z5'] = True
            m.fixed['Z6s'] = True
            m.fixed['Z7'] = True
            m.fixed['Z8'] = True
            m.fixed['Z9'] = True

        m.limits = limits
        m.strategy = 0
        m.tol = 0.01
        m.errordef = Minuit.LIKELIHOOD
        # vals = None
        # errs = None
        # for _ in range(500):
        #     m.migrad()
        #     if m.fmin.fval < current:
        #         m.hesse()
        #         current = m.fmin.fval
        #         vals = m.values
        #         errs = m.errors
        #         print(f'found min: {m.fmin.fval}')
        #     else:
        #         print(f'no min: {m.fmin.fval}')
        #     perturbed = trueValues + np.random.normal(0, 1 * np.abs(trueValues + 1e-8))
        #     split1 = len(pars[0])
        #     split2 = split1 + len(pars[1])
        #     pars = (perturbed[:split1], perturbed[split1:split2], perturbed[-1])
        m.migrad()
        m.hesse()
        print('number: ', m.nfcn)
        print(m.fmin.fval)
        print(m.fmin)
        if not m.fmin.is_valid or not m.fmin.has_accurate_covar or m.fmin.has_parameters_at_limit or m.fmin.hesse_failed:
            # print(m.fmin)
            # print(m.params)
            # print(f"Failed fit #{i} in {time.time() - tStart:.3f} seconds")
            failedCounter += 1
            continue

        vals = np.array(m.values)
        errs = np.array(m.errors)
        # normB, normBBar, arr = helper2(*vals[:-5])
        # old = -jnp.log(likelihood2(arr, vals[-5:-1], vals[-1], *data, normB, normBBar))
        #
        # nll2 = phiNLL(data[2], data[5], data[6], data[7], data[8])
        # m = Minuit(nll2, *vals[:-5], *vals[-5:-1], vals[-1])
        #
        # limits = [(-10 - 4*abs(param), 10 + 4*abs(param)) for param in trueValues]
        # limits[generation.getFitParamNames(True).index("$f$")] = (0, 1)
        # limits[generation.getFitParamNames(True).index("$c_0$")] = (0, 1)
        # m.limits = limits
        # m.strategy = 2
        # m.errordef = Minuit.LIKELIHOOD
        #
        # for p in m.fixed.to_dict():
        #     if p not in ['K3', 'K9']:
        #         m.fixed[p] = True
        # print(m.values)
        # print(nll(*vals[:-5], *vals[-5:-1], vals[-1]))
        #
        # m.migrad()
        # vals = np.array(m.values)
        #
        # print('after', nll(*vals[:-5], *vals[-5:-1], vals[-1]))
        #
        # print(m.values)
        # print(m.fmin)
        # print()
        # normB, normBBar, arr = helper2(*vals[:-5])
        #
        # new = -jnp.log(likelihood2(arr, vals[-5:-1], vals[-1], *data, normB, normBBar))
        # p = old - new

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # def plot(x, y, w, ylabel, scale):
        #     plt.figure()
        #
        #     counts, _, yedge , _= plt.hist2d(x, y, bins=20, weights=w, cmap='viridis', vmin=-scale, vmax=scale)
        #     print(np.min(counts), np.max(counts), ylabel)
        #     ax = plt.gca()
        #     divider = make_axes_locatable(ax)
        #     pullAxis = divider.append_axes('left', 1.5, pad=0.0)
        #     pullAxis.step( np.sum(counts, axis=0),  yedge[:-1])
        #
        #     plt.colorbar()
        #     ax.set_title(f'run {i}')
        #     plt.xlabel('phi')
        #     plt.ylabel(ylabel)
        #     plt.tight_layout()
        #     plt.show()
        #
        # plot(data[2], data[0], p, 'costhetaK', 5)
        # plot(data[2], data[1], p, 'costhetaL', 5)
        # plot(data[2], data[3], p, 't', 10)
        # plot(data[3], data[4], p, 'mass', 3)
        #
        #

        values[i, :] = vals
        errors[i, :] = errs
        pulls[i, :] = (vals - trueValues) / errs
        def getCov(matrix, indices):
            return matrix[indices][:,indices]

        deltacov = getCov(np.array(m.covariance), indices)
        H3 = vals[indexH3]
        H9 = vals[indexH9]
        K3 = vals[indexK3]
        K9 = vals[indexK9]

        (h3, h9, k3, k9) = uncertainties.correlated_values([H3, H9, K3, K9], deltacov)
        A = k9*float(coshIntegral) - h9*float(sinhIntegral)
        B = k3*float(coshIntegral) - h3*float(sinhIntegral)

        delta = umath.atan((B/A))
        deltavals.append(delta.nominal_value)
        deltaerrs.append(delta.std_dev)

        if i < 10:
            savedData.append(data)
        print(f"Performed fit #{i} in {time.time() - tStart:.3f} seconds")
        i += 1
    if failedCounter > 0:
        print(f"Failed {failedCounter}/{nToys} fits")

    deltavals = jnp.array(deltavals)
    deltaerrs = jnp.array(deltaerrs)
    deltapulls = (deltavals - trueDelta) / deltaerrs
    def pl(vals, values):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        results, errors = tools.fitGaussian(vals)
        nbins = 100
        if values:
            space = np.linspace(
                results[0] - 5 * results[1], results[0] + 5 * results[1], nbins
            )
        else:
            space = np.linspace(-5, 5, nbins)
        ax.hist(vals, bins=space, color="cyan")
        ax.plot(
            space,
            tools.gaussDistribution(space, *results)
            * len(vals)
            * (space[1] - space[0]),
            )
        if values:
            ax.axvline(trueDelta, color="red", linewidth=1)
        ax.set_title(f'phase shift {'pulls' if not values else 'values'}')
        text = (
            rf"$\mu$: ${float(results[0]):.3f}\pm{float(errors[0]):.3f}$"
            "\n"
            rf"$\sigma$: ${float(results[1]):.3f}\pm{float(errors[1]):.3f}$"
        )
        at = AnchoredText(
            text, loc="upper right", prop=dict(size=9), frameon=True
        )
        at.patch.set_boxstyle("round,pad=0.3")
        at.patch.set_facecolor("white")
        at.patch.set_edgecolor("gray")
        at.patch.set_alpha(0.6)
        ax.add_artist(at)

        plt.show()
    pl(deltavals, True)
    pl(deltapulls, False)

    raise Exception

    return paramNames, pulls, values, savedData


if __name__ == "__main__":

    from plot import plot
    nToys = 10
    B0proportion = 0.5
    nEvents = int(generation.q2_yields[2][2] / generation.trueValues["$f$"])
    directory = f'testing/{'theory'}_{generation.q2_range_names[2]}_{generation.integratedLuminosities[2]}'

    wSS = generation.tagging['badTagging']['wSS']
    wOS = generation.tagging['badTagging']['wOS']
    effSS = generation.tagging['badTagging']['effSS']
    effOS = generation.tagging['badTagging']['effOS']
    names, pulls, values, savedData = run(nEvents, 100, effSS, effOS, wSS, wOS, True, 2, 'theory', B0proportion=B0proportion)
    plot(names, pulls, values, savedData, nEvents, nToys, directory, 2, 2, 'theory', 'badTagging', B0proportion=B0proportion)

    # plotProjectionSummary(generation.getAllSignalParamsFromMassless(*values[0][:-5]), values[0][-5:-1], values[0][-1], savedData[0], directory, index=0, B0proportion=B0proportion, tagging=True)

    raise Exception

    for generationName in ['theory', 'alternative']:
        for q2_range in range(2,3):
            for luminosity in range(2,3):
                nEvents = int(generation.q2_yields[luminosity][q2_range] / generation.trueValues["$f$"])
                for tagging in ['badTagging', 'goodTagging']:
                    directory = f'{tagging}/{generationName}_{generation.q2_range_names[q2_range]}_{generation.integratedLuminosities[luminosity]}'
                    wSS = generation.tagging[tagging]['wSS']
                    wOS = generation.tagging[tagging]['wOS']
                    effSS = generation.tagging[tagging]['effSS']
                    effOS = generation.tagging[tagging]['effOS']
                    names, pulls, values, savedData = run(nEvents, nToys, effSS, effOS, wSS, wOS, tagging != 'untagged', q2_range, generationName, B0proportion=B0proportion)
                    plot(names, pulls, values, savedData, nEvents, nToys, directory, q2_range, luminosity, generationName, tagging, B0proportion=B0proportion)
                    print(f'Completed {directory}')