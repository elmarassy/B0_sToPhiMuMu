import os
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
def individualTimeDependent(normalization, sign, cosThetaL, cosThetaK, phi, t, mass,
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

    return (9.0 / 64.0 / jnp.pi / normalization) * acceptance(t) * (
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
    return (1./8) * (
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
    sin = sinIntegral
    sinh = sinhIntegral
    cos = cosIntegral
    cosh = coshIntegral
    return (3./(16.*norm)) * (-Z1c*sin - 2*Z1s*sin + Z2c*sin + 2*Z2s*sin - H1c*sinh -
                              2*H1s*sinh + H2c*sinh + 2*H2s*sinh + 2*cos*K6s*cosThetaL - 2*Z6s*sin*cosThetaL -
                              2*H6s*sinh*cosThetaL - 2*Z2c*sin*cosThetaL**2 - 4*Z2s*sin*cosThetaL**2 - 2*H2c*sinh*cosThetaL**2 -
                              4*H2s*sinh*cosThetaL**2 + cos*W1c + 2*cos*W1s - cos*W2c + 2*cos*cosThetaL**2*W2c -
                              2*cos*W2s + 4*cos*cosThetaL**2*W2s + cosh*(K1c + 2*K1s - K2c - 2*K2s + 2*K2c*cosThetaL**2 + 4*K2s*cosThetaL**2 + 2*cosThetaL*W6s))


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
    return (3./16.) * (
        ((3*K1c-K2c)*coshIntegral - (3*H1c-H2c)*sinhIntegral + sign*((3*W1c-W2c)*cosIntegral - (3*Z1c-Z2c)*sinIntegral)) * cosThetaK**2 +
        ((3*K1s-K2s)*coshIntegral - (3*H1s-H2s)*sinhIntegral + sign*((3*W1s-W2s)*cosIntegral - (3*Z1s-Z2s)*sinIntegral)) * (1 - cosThetaK**2)
    ) / norm


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

    return (1./(16*jnp.pi*norm)) * (
        (3*K1c + 6*K1s - K2c - 2*K2s)*coshIntegral - (3*H1c + 6*H1s - H2c - 2*H2s)*sinhIntegral +
        sign*((3*W1c + 6*W1s - W2c - 2*W2s)*cosIntegral - (3*Z1c + 6*Z1s - Z2c - 2*Z2s)*sinIntegral) +
        4*(K3*coshIntegral - H3*sinhIntegral + sign*(W3*cosIntegral - Z3*sinIntegral))*jnp.cos(2*phi) +
        4*(K9*coshIntegral - H9*sinhIntegral + sign*(W3*cosIntegral - Z9*sinIntegral))*jnp.sin(2*phi)
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

    return (1./8.) * acceptance(t) * (
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
def likelihood(signalParams, backgroundParams, f, cosThetaL, cosThetaK, phi, time, mass, qSS, qOS, wSS, wOS, normB, normBBar):
    background = backgroundDistribution(time, mass, *backgroundParams) / integralBackground(*backgroundParams)
    weightB = (1 + qSS * (1 - 2 * wSS)) * (1 + qOS * (1 - 2 * wOS)) / 4
    weightBBar = (1 - qSS * (1 - 2 * wSS)) * (1 - qOS * (1 - 2 * wOS)) / 4
    b = individualTimeDependent(1, 1, cosThetaL, cosThetaK, phi, time, mass, *signalParams)
    bBar = individualTimeDependent(1, -1, cosThetaL, cosThetaK, phi, time, mass, *signalParams)
    pdf = (
             weightB *
             (f*b/normB + (1-f)*background) +
             weightBBar *
             (f*bBar/normBBar + (1-f)*background))
    weightUntagged = 1 - weightB - weightBBar
    pdf2 = weightUntagged * (f*(b+bBar)/(normB+normBBar) + (1-f)*background)
    return -jnp.sum(jnp.log(pdf))#-jnp.sum(jnp.log(pdf2))


@jit
def helper2(K1c, K3, K4, K5, K6s, K7, K8, K9,
            W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
            H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
            Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):

    K1s = (3.0/4.0) * (1 - y*y - K1c + y*H1c) + y*H1s
    K2s = (1.0/4.0) * (1 - y*y - K1c + y*H1c) + y*H1s/3.0

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
    return lambda cosThetaL, cosThetaK, phi, t, mass, qSS, qOS, wSS, wOS: (
        likelihood(arr, (c0, c1, c2, kM), f, cosThetaL, cosThetaK, phi, t, mass, qSS, qOS, wSS, wOS, normB, normBBar))

@jit
def value(params, cosThetaL, cosThetaK, phi, time, mass, qSS, qOS, wSS, wOS):
    return masslessNLLHelper(*params)(cosThetaL, cosThetaK, phi, time, mass, qSS, qOS, wSS, wOS)

gradHelper = jax.grad(value, argnums=0)
class MasslessNLL:

    def __init__(self, cosThetaL, cosThetaK, phi, time, mass, qSS, qOS, wSS, wOS):
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
                                       c0, c1, c2, kM, f)(self.cosThetaL, self.cosThetaK, self.phi, self.time, self.mass, self.qSS, self.qOS, self.wSS, self.wOS)
        if jnp.isfinite(likelihood):
            return float(likelihood)
        return 1e12

    @property
    def grad(self):
        return lambda *params: self.gradientHelper(params, self.cosThetaL, self.cosThetaK, self.phi, self.time, self.mass, self.qSS, self.qOS, self.wSS, self.wOS)

@functools.partial(jit, static_argnums=(1,))
def generate(key, N, signalParams, backgroundParams, f, B0proportion, effSS, effOS, mistagSS, mistagOS):
    keys = random.split(key, 17)
    mul = 1000

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
    massSignal = random.uniform(keys[10], shape=(mul*N,), minval=massRange[0], maxval=massRange[1])
    normB = integrateIndividual(1, *signalParams)
    normBBar = integrateIndividual(-1, *signalParams)
    norm = jnp.where(trueTag, normB, normBBar)
    signalProbabilities = individualTimeDependent(norm, trueTagValue, cosThetaLSignal, cosThetaKSignal, phiSignal, tSignal, massSignal, *signalParams)
    signalProbabilities /= signalProbabilities.max()

    cosThetaLBackground = random.uniform(keys[11], shape=(mul*N,), minval=-1.0, maxval=1.0)
    cosThetaKBackground = random.uniform(keys[12], shape=(mul*N,), minval=-1.0, maxval=1.0)
    phiBackground = random.uniform(keys[13], shape=(mul*N,), minval=-jnp.pi, maxval=jnp.pi)
    tBackground = random.uniform(keys[14], shape=(mul*N,), minval=timeRange[0], maxval=timeRange[1])
    massBackground = random.uniform(keys[15], shape=(mul*N,), minval = massRange[0], maxval=massRange[1])
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

    return cosThetaL, cosThetaK, phi, t, mass, tagChoiceSS, tagChoiceOS, jnp.full(N, mistagSS), jnp.full(N, mistagOS)


def run(nEvents, nToys, effSS, effOS, wSS, wOS, saveProjectionValue=4, massless=True, B0proportion=0.5, untagged=False):
    key = jax.random.key(0)

    if untagged:
        effSS = 0
        effOS = 0
    paramNames = generation.getFitParamNames(massless)
    pars = generation.getFitParams(massless)
    trueValues = np.array(pars[0] + pars[1] + [pars[2]])
    savedData = []
    values = np.empty(shape=(nToys, len(paramNames)))
    errors = np.empty(shape=(nToys, len(paramNames)))
    pulls = np.empty(shape=(nToys, len(paramNames)))

    i = 0
    failedCounter = 0
    while i < nToys:
        key, key1 = jax.random.split(key)
        tStart = time.time()
        pars = generation.getFitParams(massless)
        data = generate(key1, nEvents, generation.getAllSignalParamsFromMassless(*pars[0]), pars[1], pars[2], B0proportion, effSS, effOS, wSS, wOS)
        nll = MasslessNLL(*data)
        m = Minuit(nll, *pars[0], *pars[1], pars[2])

        limits = [(-1 - 4*abs(param), 1 + 4*abs(param)) for param in trueValues]
        limits[generation.getFitParamNames(True).index("$f$")] = (0, 1)
        limits[generation.getFitParamNames(True).index("$c_0$")] = (0, 1)
        if untagged:
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
        m.errordef = Minuit.LIKELIHOOD
        m.migrad()
        m.hesse()
        print('number: ', m.nfcn)
        if not m.fmin.is_valid or not m.fmin.has_accurate_covar or m.fmin.has_parameters_at_limit or m.fmin.hesse_failed:
            print(m.fmin)
            print(m.params)
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
    if failedCounter > 0:
        print(f"Failed {failedCounter}/{nToys} fits")
    return paramNames, pulls, values, savedData

def main():

    nEvents = 6000
    nToys = 3
    wSS = generation.trueValues['wSS']
    wOS = generation.trueValues['wOS']
    effSS = generation.trueValues['effSS']
    effOS = generation.trueValues['effOS']

    names, pulls, values, savedData = run(nEvents, nToys, effSS, effOS, wSS, wOS, untagged=False)
    plot.plot(names, pulls, values, savedData, nEvents, nToys, 'plots')


    # plot.project(saveProjectionsValue, [], [], [], effSS, effOS, wSS, wOS, "plots")
    #
    # plot.plotProjectionSummary(saveProjectionsValue, pulls, values, savedData, effSS, effOS, wSS, wOS, "plots")
    # plot.plotSummary(names, pulls, True, False, "plots")
    # plot.plotSummary(names, values, True, True, "plots", generation.getFitParams(massless))


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    import plot
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('time')
    # stats.print_stats(100)